import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io
import plotly.graph_objects as go

register_page(__name__, path='/interpolation', name='Interpolation')

# Styling
card_style = {
    'border': '1px solid #ccc',
    'padding': '20px',
    'borderRadius': '8px',
    'marginBottom': '20px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
    'backgroundColor': '#fafafa'
}
button_style = {
    'backgroundColor': '#2196F3',
    'color': 'white',
    'padding': '10px 20px',
    'border': 'none',
    'borderRadius': '5px',
    'cursor': 'pointer',
    'marginRight': '10px'
}

layout = html.Div([
    html.H2('Interpolation Techniques', style={'textAlign':'center', 'marginBottom':'30px', 'fontSize': '2.5rem'}),

    html.Div([
        html.H4('Upload Data File'),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel File')]),
            style={
                'width':'100%','height':'60px','lineHeight':'60px',
                'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                'textAlign':'center','backgroundColor':'#fafafa'
            }, multiple=False
        ),
        html.Div(id='file-info', style={'marginTop':'10px'})
    ], style=card_style),

    html.Div([
        html.Div([html.Label('Time Column:'), dcc.Dropdown(id='time-column')], style={'flex':1, 'marginRight':'10px'}),
        html.Div([html.Label('Value Column:'), dcc.Dropdown(id='value-column')], style={'flex':1, 'marginRight':'10px'}),
        html.Div([html.Label('Points per Interval:'), dcc.Input(id='interp-freq', type='number', value=1, min=1, step=1)], style={'flex':1})
    ], style={**card_style, 'display':'flex'}),

    html.Div([
        html.Button('Interpolate Full Series', id='interp-btn', style=button_style),
        html.Button('Replace Selected Segment', id='apply-btn', style=button_style),
        html.Button('Download Data', id='download-btn', style=button_style)
    ], style={'textAlign':'center', 'marginBottom':'20px'}),

    dcc.Store(id='selected-points', data=[]),
    dcc.Store(id='modified-data', data=None),

    dcc.Loading(
        children=[dcc.Graph(id='interp-graph')],
        type='default'
    ),

    html.Div(id='selected-points-display', style={
        'whiteSpace': 'pre-wrap', 'marginTop': '20px',
        'borderTop': '1px solid #ccc', 'paddingTop': '10px'
    })
], style={
    'maxWidth':'900px', 'margin':'auto', 'padding':'20px',
    'fontFamily':'Roboto, sans-serif', 'backgroundColor':'white'
})


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        pass
    return pd.DataFrame()

@dash.callback(
    Output('file-info', 'children'),
    Output('time-column', 'options'),
    Output('value-column', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def update_dropdowns(contents, filename):
    df = parse_contents(contents, filename)
    if df.empty:
        return '❌ Failed to load file.', [], []
    options = [{'label': c, 'value': c} for c in df.columns]
    return f'✅ Loaded: {filename}', options, options

@dash.callback(
    Output('selected-points', 'data'),
    Input('interp-graph', 'clickData'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)
def capture_point(clickData, stored):
    if not clickData:
        return stored
    x = clickData['points'][0]['x']
    if x in stored:
        stored.remove(x)
    else:
        stored.append(x)
    return stored

@dash.callback(
    Output('interp-graph', 'figure'),
    Input('interp-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('time-column', 'value'),
    State('value-column', 'value'),
    State('interp-freq', 'value'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)
def update_graph(n_clicks, contents, filename, time_col, value_col, freq, selected):
    df = parse_contents(contents, filename)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    times = df[time_col]
    values = df[value_col].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode='markers+lines', name='Original'))
    if selected:
        sel_times = pd.to_datetime(selected)
        sel_vals = df.loc[df[time_col].isin(sel_times), value_col]
        fig.add_trace(go.Scatter(x=sel_times, y=sel_vals, mode='markers', marker={'color':'red','size':10}, name='Selected'))
    t0 = (df[time_col] - df[time_col].iloc[0]).dt.total_seconds().values
    y0 = values.values
    new_t, new_y = [], []
    for i in range(len(t0)-1):
        a, b = t0[i], t0[i+1]
        ya, yb = y0[i], y0[i+1]
        new_t.append(a); new_y.append(ya)
        for s in np.linspace(0,1,freq+2)[1:-1]:
            new_t.append(a + s*(b-a)); new_y.append(ya + s*(yb-ya))
    new_t.append(t0[-1]); new_y.append(y0[-1])
    interp_times = df[time_col].iloc[0] + pd.to_timedelta(new_t, unit='s')
    fig.add_trace(go.Scatter(x=interp_times, y=new_y, mode='lines', name='Interpolated', line={'dash':'dash'}))
    fig.update_layout(
        title='Interpolation', xaxis_title='Time', yaxis_title=value_col,
        plot_bgcolor='white',
        template='plotly_white'
    )
    return fig

@dash.callback(
    Output('selected-points-display', 'children'),
    Input('selected-points', 'data')
)
def display_selected(selected):
    if not selected:
        return 'No points selected.'
    return 'Selected times:\n' + '\n'.join([str(pd.to_datetime(x)) for x in selected])

@dash.callback(
    Output('modified-data', 'data'),
    Input('apply-btn', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('time-column', 'value'),
    State('value-column', 'value'),
    State('interp-freq', 'value'),
    State('selected-points', 'data'),
    prevent_initial_call=True
)
def apply_interpolation(n_clicks, contents, filename, time_col, value_col, freq, selected):
    df = parse_contents(contents, filename)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    df['interpolated'] = False
    if selected and len(selected) >= 2:
        sel = sorted(pd.to_datetime(selected))[:2]
        start, end = sel
        before = df[df[time_col] < start]
        after = df[df[time_col] > end]
        seg = df[(df[time_col] >= start) & (df[time_col] <= end)].reset_index(drop=True)
        t0 = (seg[time_col] - seg[time_col].iloc[0]).dt.total_seconds().values
        y0 = seg[value_col].astype(float).values
        new_t, new_y = [], []
        for i in range(len(t0)-1):
            a, b = t0[i], t0[i+1]
            ya, yb = y0[i], y0[i+1]
            new_t.append(a); new_y.append(ya)
            for s in np.linspace(0,1,freq+2)[1:-1]:
                new_t.append(a + s*(b-a)); new_y.append(ya + s*(yb-ya))
        new_t.append(t0[-1]); new_y.append(y0[-1])
        interp_times = seg[time_col].iloc[0] + pd.to_timedelta(new_t, unit='s')
        interp_df = pd.DataFrame({time_col: interp_times, value_col: new_y})
        interp_df['interpolated'] = True
        result = pd.concat([before, interp_df, after]).sort_values(time_col).reset_index(drop=True)
    else:
        result = df.copy()
    return result.to_dict('records')

@dash.callback(
    Output('download-dataframe', 'data'),
    Input('download-btn', 'n_clicks'),
    State('modified-data', 'data'),
    prevent_initial_call=True
)
def download_data(n_clicks, data):
    df_mod = pd.DataFrame(data)
    return dcc.send_data_frame(df_mod.to_csv, 'interpolated_data.csv', index=False)