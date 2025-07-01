import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

# Register this page with Dash
register_page(__name__, path='/exposure-timing', name='Exposure Timing')

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
    html.Div(id='alert-box', style=alert_style),
    
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
        html.H1('Exposure Timing Optimization', style={'textAlign':'center','marginBottom':'40px'}),
        
        # File upload section
        html.Div([
            html.H4('Upload Sensor Data'),
            dcc.Upload(
                id='et-upload',
                children=html.Div(['📁 Drag & Drop or ', html.A('Select CSV File')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','marginBottom':'10px','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='et-file-info', style={'marginBottom':'20px'})
        ], style=card_style),
        
        # Column selection
        html.Div([
            html.Div([
                html.Label('Time Column:'),
                dcc.Dropdown(id='et-time-col', options=[], placeholder='Select time column')
            ], style={'flex':1, 'marginRight':'10px'}),
            html.Div([
                html.Label('Value Column:'),
                dcc.Dropdown(id='et-value-col', options=[], placeholder='Select value column')
            ], style={'flex':1})
        ], style={'display':'flex', 'marginBottom':'20px'}),
        
        # Parameter configuration
        html.Div([
            html.Label('Target Value:'),
            dcc.Input(id='et-target', type='number', placeholder='Enter target value', style={'width':'100%'})
        ], style={'marginBottom':'20px'}),
        
        # Compute button
        html.Div(html.Button('Optimize Exposure Timing', id='et-btn', style=button_style), style={'textAlign':'center','marginBottom':'40px'}),
        
        # Results display
        dcc.Loading(
            children=[
                dcc.Graph(id='et-graph'),
                html.Div(id='et-output', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'})
            ], type='default'
        ),
        
        # Computation time
        html.Div(id='computation-time', style={'marginTop': '10px', 'fontStyle': 'italic'})
    ], style={
        'marginLeft':'240px','padding':'20px','paddingTop':'100px',
        'fontFamily':'Roboto, sans-serif'
    }),
    
    # Data stores
    dcc.Store(id='et-data-store')
])

# parse_contents helper
def parse_contents(contents, filename):
    """
    Parse the uploaded file and return a pandas DataFrame.
    
    Parameters:
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    
    Returns:
    pd.DataFrame: The parsed DataFrame.
    """
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

# populate dropdowns
@dash.callback(
    Output('et-file-info','children'),
    Output('et-time-col','options'),
    Output('et-value-col','options'),
    Input('et-upload','contents'),
    State('et-upload','filename'),
    prevent_initial_call=True
)
def update_et_dropdowns(contents, filename):
    """
    Update the dropdown options based on the uploaded file.
    
    Parameters:
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    
    Returns:
    str: The file info message.
    list: The time column options.
    list: The value column options.
    """
    df = parse_contents(contents, filename)
    if df.empty:
        return '❌ Failed to load file.', [], []
    opts = [{'label':c,'value':c} for c in df.columns]
    return f'✅ Loaded: {filename}', opts, opts

# find exposure time
@dash.callback(
    Output('et-graph','figure'),
    Output('et-output','children'),
    Input('et-btn','n_clicks'),
    State('et-upload','contents'),
    State('et-upload','filename'),
    State('et-time-col','value'),
    State('et-value-col','value'),
    State('et-target','value'),
    prevent_initial_call=True
)
def find_exposure(n, contents, filename, tcol, vcol, target):
    """
    Find the exposure time based on the uploaded file and selected parameters.
    
    Parameters:
    n (int): The number of clicks on the compute button.
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    tcol (str): The selected time column.
    vcol (str): The selected value column.
    target (float): The target value.
    
    Returns:
    plotly.graph_objects.Figure: The exposure time graph.
    str: The exposure time output message.
    """
    df = parse_contents(contents, filename)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).reset_index(drop=True)
    times = df[tcol]
    vals = df[vcol].astype(float)
    # compute cumulative exposure (trapezoidal)
    t_sec = (times - times.iloc[0]).dt.total_seconds().values
    E = np.zeros_like(vals)
    for i in range(1,len(vals)):
        dt = t_sec[i] - t_sec[i-1]
        E[i] = E[i-1] + (vals[i-1]+vals[i])/2* dt
    # find bracket
    diff = E - target
    sign = np.sign(diff)
    idx = np.where(sign[:-1]*sign[1:]<0)[0]
    if len(idx)==0:
        return go.Figure(), 'Target not reached.'
    i = idx[0]
    a, b = t_sec[i], t_sec[i+1]
    Ea, Eb = E[i], E[i+1]
    def g(x):
        return np.interp(x, [a,b], [Ea,Eb]) - target
    # root find
    it=0
    while (b-a)>1e-6:
        m=(a+b)/2
        if g(a)*g(m)<=0: b=m
        else: a=m
        it+=1
    root=(a+b)/2; res=g(root)
    t_root = times.iloc[0] + pd.to_timedelta(root, unit='s')
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=E, mode='markers+lines', name='Exposure'))
    fig.add_hline(y=target, line_dash='dash', annotation_text='Target')
    fig.add_trace(go.Scatter(x=[t_root], y=[target], mode='markers', marker={'color':'red','size':12}, name='Crossing'))
    fig.update_layout(title='Exposure vs Time', xaxis_title='Time', yaxis_title='Cumulative Exposure', plot_bgcolor='white', template='plotly_white')
    out = f"Target reached at {t_root.isoformat()} (iterations={it}, residual={res:.2e})"
    return fig, out
