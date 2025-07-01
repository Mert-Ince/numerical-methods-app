import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go
from scipy.integrate import simpson

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

# Register this page with Dash
register_page(__name__, path='/integration', name='Integration')

# Styles for UI components
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
alert_style = {
    'position': 'fixed',
    'top': '10px',
    'right': '10px',
    'width': '300px',
    'zIndex': '1000',
    'padding': '10px',
    'borderRadius': '5px',
    'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
}

# Main page layout
layout = html.Div([
    # Alert box for user feedback
    html.Div(id='int-alert-box', style=alert_style),
    
    # Sidebar navigation
    html.Div([
        html.H4('Tools', style={'marginBottom':'20px'}),
        dcc.Link('Threshold Crossing', href='/threshold-crossing', style=link_style),
        dcc.Link('Exposure Timing', href='/exposure-timing', style=link_style),
        dcc.Link('Trend Analysis', href='/trend-analysis', style=link_style),
        dcc.Link('Integration', href='/integration', style=link_style),
        dcc.Link('ODE Solver', href='/ode-solver', style=link_style)
    ], style=sidebar_style),
    
    # Main content area
    html.Div([
        html.H2('Numerical Integration', style={'textAlign':'center','marginBottom':'30px'}),
        
        # File upload section
        html.Div([
            html.H4('Upload Data File'),
            dcc.Upload(
                id='int-upload',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel File')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='int-file-info', style={'marginTop':'15px'})
        ], style=card_style),

        html.Div([
            html.Div([html.Label('Time Column:'), dcc.Dropdown(id='int-time-col')], style={'flex':1,'marginRight':'10px'}),
            html.Div([html.Label('Value Column:'), dcc.Dropdown(id='int-value-col')], style={'flex':1})
        ], style={**card_style,'display':'flex'}),

        html.Div([
            html.Div([html.Label('Method:'), dcc.Dropdown(
                id='int-method',
                options=[
                    {'label':'Trapezoidal','value':'trap'},
                    {'label':'Simpson','value':'simpson'}
                ], value='trap'
            )], style={'flex':1,'marginRight':'10px'}),
            html.Div([html.Label('Rolling Window (hours, 0=full):'),
                      dcc.Input(id='int-window', type='number', value=0, min=0, step=0.1)
            ], style={'flex':1})
        ], style={**card_style,'display':'flex'}),

        html.Div(html.Button('Compute Integration', id='int-btn', style=button_style),
                 style={'textAlign':'center','marginBottom':'40px'}),

        dcc.Loading(children=[
            dcc.Graph(id='int-graph'),
            dcc.Graph(id='int-window-graph')
        ], type='default'),

        html.Div(id='int-output', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'}),
        html.Div(id='int-computation-time', style={'marginTop': '10px', 'fontStyle': 'italic'})
    ], style={
        'marginLeft':'240px','padding':'20px','paddingTop':'100px',
        'fontFamily':'Roboto, sans-serif'
    })
])

# Helper function for data parsing
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

# Updated callback using global exception handler and utilities
@dash.callback(
    [Output('int-graph','figure'),
     Output('int-window-graph','figure'),
     Output('int-output','children'),
     Output('int-computation-time','children'),
     Output('int-alert-box','children')],
    [Input('int-btn','n_clicks')],
    [State('int-upload','contents'),
     State('int-upload','filename'),
     State('int-time-col','value'),
     State('int-value-col','value'),
     State('int-method','value'),
     State('int-window','value')],
    prevent_initial_call=True
)
def compute_integration(n, contents, filename, tcol, vcol, method, window):
    if n is None:
        raise dash.exceptions.PreventUpdate
    
    # Parse and validate
    df = parse_contents(contents, filename)
    
    # Validate with shared utility
    errors = validate_inputs(
        df,
        required_columns=[tcol, vcol],
        datetime_columns=[tcol],
        numeric_columns=[vcol]
    )
    
    if errors:
        return (
            go.Figure(),
            go.Figure(),
            "",
            "",
            create_alert("Validation errors: " + "; ".join(errors), 'error')
        )
    
    # Process data
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)
    times = df[tcol]
    values = df[vcol].values
    
    # Timing
    start_time = time.perf_counter()
    
    # Convert times to seconds
    t0 = times.min()
    t_sec = (times - t0).dt.total_seconds().values
    
    # Compute integral with timing
    if method == 'trap':
        integral = np.trapz(values, t_sec)
    elif method == 'simpson':
        integral = simpson(values, t_sec)
    
    # Create visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode='lines', name='Value', fill='tozeroy'))
    fig.update_layout(title='Total Integrated Exposure', xaxis_title='Time', yaxis_title=vcol, plot_bgcolor='white', template='plotly_white')
    
    # rolling window
    fig_win = go.Figure()
    if window>0:
        win_sec = window*3600
        Ewin = []
        for i, t0 in enumerate(t_sec):
            mask = (t_sec>=t0-win_sec)&(t_sec<=t0)
            if method=='trap':
                Ewin.append(np.trapz(values[mask], t_sec[mask]))
            else:
                Ewin.append(simpson(values[mask], t_sec[mask]))
        fig_win.add_trace(go.Scatter(x=times, y=Ewin, mode='lines', name=f'{window}h Window'))
        fig_win.update_layout(title='Rolling Window Exposure', xaxis_title='Time', yaxis_title='Exposure', plot_bgcolor='white', template='plotly_white')
    
    # summary
    summary = f"Total exposure = {integral/3600:.2f} (unit·hours)"
    
    computation_time = time.perf_counter() - start_time
    
    return (
        fig,
        fig_win,
        summary,
        f"Computed in {computation_time:.4f} seconds",
        create_alert(f"Success! Computed in {computation_time:.4f}s", 'success')
    )

# populate dropdowns
@dash.callback(
    Output('int-file-info','children'),
    Output('int-time-col','options'),
    Output('int-value-col','options'),
    Input('int-upload','contents'),
    State('int-upload','filename'),
    prevent_initial_call=True
)
def update_int_dropdowns(contents, filename):
    df = parse_contents(contents, filename)
    if df.empty:
        return '❌ Failed to load file.',[],[]
    opts = [{'label':c,'value':c} for c in df.columns]
    return f'✅ Loaded: {filename}', opts, opts
