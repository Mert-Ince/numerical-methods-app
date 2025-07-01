import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

register_page(__name__, path='/threshold-crossing', name='Threshold Crossing')

# Styles
card_style = {
    'border': '1px solid #ccc',
    'padding': '20px',
    'borderRadius': '8px',
    'marginBottom': '30px',
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
    'marginTop': '10px'
}
sidebar_style = {
    'position': 'fixed',
    'top': '60px',
    'left': '0',
    'width': '200px',
    'height': 'calc(100% - 80px)',
    'color':'#2c3e50',
    'background': 'linear-gradient(90deg, #f5f7fa, #c3cfe2)',
    'padding': '20px',
    'boxShadow': '2px 0 5px rgba(0,0,0,0.1)'
}
link_style = {
    'display': 'block',
    'marginBottom': '15px',
    'color': '#2c3e50',
    'textDecoration': 'none',
    'fontWeight': 'bold'
}

layout = html.Div([
    # Sidebar
    html.Div([
        html.H4('Tools', style={'marginBottom':'20px'}),
        dcc.Link('Threshold Crossing', href='/threshold-crossing', style=link_style),
        dcc.Link('Exposure Timing', href='/exposure-timing', style=link_style),
        dcc.Link('Trend Analysis', href='/trend-analysis', style=link_style),
        dcc.Link('Integration', href='/integration', style=link_style),
        dcc.Link('ODE Solver', href='/ode-solver', style=link_style)
    ], style=sidebar_style),

    # Main content
    html.Div([
        html.H2('Precise Threshold Crossing', style={'textAlign': 'center', 'marginBottom': '30px'}),

        html.Div([
            html.H4('Upload Data File'),
            dcc.Upload(
                id='tc-upload',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel File')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='tc-file-info', style={'marginTop':'15px'})
        ], style=card_style),

        html.Div([
            html.Div([html.Label('Time Column:'), dcc.Dropdown(id='tc-time-col')], style={'flex':1, 'marginRight':'10px'}),
            html.Div([html.Label('Value Column:'), dcc.Dropdown(id='tc-value-col')], style={'flex':1})
        ], style={**card_style, 'display':'flex'}),

        html.Div([
            html.Div([html.Label('Threshold:'), dcc.Input(id='tc-threshold', type='number', value=100)], style={'flex':1, 'marginRight':'10px'}),
            html.Div([html.Label('Method:'), dcc.RadioItems(
                id='tc-method',
                options=[
                    {'label':'Bisection','value':'bisection'},
                    {'label':'Secant','value':'secant'}
                ],
                value='bisection',
                labelStyle={'display':'inline-block','marginRight':'15px'}
            )], style={'flex':2})
        ], style={**card_style, 'display':'flex'}),

        html.Div([
            html.Label('Tolerance (seconds):'),
            dcc.Input(id='tc-tol', type='number', value=1, step=0.1)
        ], style={**card_style, 'width':'200px'}),

        html.Div(html.Button('Find Crossing', id='tc-btn', style=button_style), style={'textAlign':'center','marginBottom':'40px'}),

        dcc.Loading(
            children=[dcc.Graph(id='tc-graph')],
            type='default'
        ),

        html.Div(id='tc-output', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'}),
        html.Div(id='tc-computation-time', style={'marginTop':'10px','textAlign':'center'}),
        html.Div(id='tc-alert-box', style={'marginTop':'10px','textAlign':'center'})
    ], style={
        'marginLeft':'240px',
        'padding':'20px',
        'paddingTop':'100px',  # space for navbar
        'fontFamily':'Roboto, sans-serif'
    })
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith(('.xls','.xlsx')):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        pass
    return pd.DataFrame()

@dash.callback(
    Output('tc-file-info','children'),
    Output('tc-time-col','options'),
    Output('tc-value-col','options'),
    Input('tc-upload','contents'),
    State('tc-upload','filename'),
    prevent_initial_call=True
)
def update_tc_dropdowns(contents, filename):
    df = parse_contents(contents, filename)
    if df.empty:
        return '❌ Failed to load file.', [], []
    opts = [{'label':c,'value':c} for c in df.columns]
    return f'✅ Loaded: {filename}', opts, opts

@dash.callback(
    Output('tc-graph','figure'),
    Output('tc-output','children'),
    Output('tc-computation-time','children'),
    Output('tc-alert-box','children'),
    Input('tc-btn','n_clicks'),
    [State('tc-upload','contents'),
     State('tc-upload','filename'),
     State('tc-time-col','value'),
     State('tc-value-col','value'),
     State('tc-threshold','value'),
     State('tc-method','value'),
     State('tc-tol','value')],
    prevent_initial_call=True
)
def compute_crossings(n, contents, filename, tcol, vcol, threshold, method, tol):
    if n is None:
        raise dash.exceptions.PreventUpdate
    
    # Parse and validate
    df = parse_contents(contents, filename)
    
    # Validate inputs
    errors = validate_inputs(
        df,
        required_columns=[tcol, vcol],
        datetime_columns=[tcol],
        numeric_columns=[vcol]
    )
    
    if errors:
        return go.Figure(), "", "", create_alert("Validation errors: " + "; ".join(errors), 'error')
    
    # Timing
    start_time = time.perf_counter()
    
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).reset_index(drop=True)
    times = df[tcol]
    vals = df[vcol].astype(float)
    diff = vals - threshold
    sign = np.sign(diff)
    idx = np.where(sign[:-1]*sign[1:]<0)[0]
    if len(idx)==0:
        return go.Figure(), 'No crossing detected.', "", create_alert("No crossing detected.", 'error')
    i = idx[0]
    t0, t1 = times.iloc[i], times.iloc[i+1]
    f0, f1 = vals.iloc[i], vals.iloc[i+1]
    t_sec = (times - times.iloc[0]).dt.total_seconds().values
    f = vals.values
    a, b = t_sec[i], t_sec[i+1]
    def g(x):
        return np.interp(x, [a,b], [f0,f1]) - threshold
    if method=='bisection':
        it=0
        while (b-a)>tol:
            m=(a+b)/2
            if g(a)*g(m)<=0: b=m
            else: a=m
            it+=1
        root=(a+b)/2
        res=g(root)
    else:
        x0, x1 = a, b
        it=0
        while True:
            f0n, f1n = g(x0), g(x1)
            x2 = x1 - f1n*(x1-x0)/(f1n-f0n)
            x0, x1 = x1, x2
            it+=1
            if abs(g(x1))<tol: break
        root=x1; res=g(root)
    t_root = times.iloc[0] + pd.to_timedelta(root, unit='s')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=vals, mode='markers+lines', name='Data'))
    fig.add_hline(y=threshold, line_dash='dash', annotation_text='Threshold')
    fig.add_trace(go.Scatter(x=[t_root], y=[threshold], mode='markers', marker={'color':'red','size':12}, name='Crossing'))
    fig.update_layout(title='Threshold Crossing', xaxis_title='Time', yaxis_title=vcol, plot_bgcolor='white', template='plotly_white')
    computation_time = time.perf_counter() - start_time
    out = f"Crossing at {t_root.isoformat()} ({method}, iterations={it}, residual={res:.2e})"
    return (
        fig,
        out,
        f"Computed in {computation_time:.4f} seconds",
        create_alert(f"Success! Computed in {computation_time:.4f}s", 'success')
    )
