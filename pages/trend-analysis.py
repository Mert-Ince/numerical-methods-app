import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go
from scipy.stats import linregress

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

# Register this page with Dash
register_page(__name__, path='/trend-analysis', name='Trend Analysis')

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

# Main page layout
layout = html.Div([
    html.Div([
        html.H4('Tools', style={'marginBottom':'20px'}),
        dcc.Link('Threshold Crossing', href='/threshold-crossing', style=link_style),
        dcc.Link('Exposure Timing', href='/exposure-timing', style=link_style),
        dcc.Link('Trend Analysis', href='/trend-analysis', style=link_style),
        dcc.Link('Integration', href='/integration', style=link_style),
        dcc.Link('ODE Solver', href='/ode-solver', style=link_style)
    ], style=sidebar_style),

    html.Div([
        html.H2('Trend Analysis via Numerical Differentiation', style={'textAlign': 'center', 'marginBottom': '30px'}),

        # File upload section
        html.Div([
            html.H4('Upload Data File'),
            dcc.Upload(
                id='ta-upload',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel File')]),
                style={
                    'width':'100%', 'height':'60px', 'lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='ta-file-info', style={'marginTop':'15px'})
        ], style=card_style),

        # Column selection
        html.Div([
            html.Div([html.Label('Time Column:'), dcc.Dropdown(id='ta-time-col')], style={'flex':1, 'marginRight':'10px'}),
            html.Div([html.Label('Value Column:'), dcc.Dropdown(id='ta-value-col')], style={'flex':1})
        ], style={**card_style, 'display':'flex'}),

        # Method selection
        html.Div([
            html.Div([html.Label('Method:'), dcc.Dropdown(
                id='ta-method',
                options=[
                    {'label':'Forward Difference','value':'forward'},
                    {'label':'Backward Difference','value':'backward'},
                    {'label':'Central Difference','value':'central'}
                ], value='central'
            )], style={'flex':1, 'marginRight':'10px'}),
            html.Div([html.Label('Step Size (points):'), dcc.Input(id='ta-step', type='number', value=1, min=1, step=1)], style={'flex':1})
        ], style={**card_style, 'display':'flex'}),

        # Parameter configuration
        html.Div([
            html.Label('Trend Threshold (slope units/time):'),
            dcc.Input(id='ta-threshold', type='number', value=1.0, step=0.1)
        ], style={**card_style, 'width':'300px'}),

        # Compute button
        html.Div(html.Button('Compute Trends', id='ta-btn', style=button_style), style={'textAlign':'center','marginBottom':'40px'}),

        # Results display
        dcc.Loading(
            children=[dcc.Graph(id='ta-graph')], type='default'
        ),

        # Summary statistics
        html.Div(id='ta-summary', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'}),
        html.Div(id='ta-computation-time', style={'marginTop':'10px','textAlign':'center'}),
        html.Div(id='ta-alert-box', style={'marginTop':'10px','textAlign':'center'})
    ], style={
        'marginLeft':'240px',
        'padding':'20px',
        'paddingTop':'100px',
        'fontFamily':'Roboto, sans-serif'
    })
])

# Helper function to load file
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

# Populate dropdowns
@dash.callback(
    Output('ta-file-info','children'),
    Output('ta-time-col','options'),
    Output('ta-value-col','options'),
    Input('ta-upload','contents'),
    State('ta-upload','filename'),
    prevent_initial_call=True
)
def update_ta_dropdowns(contents, filename):
    """
    Update the file info and dropdown options when a file is uploaded.
    
    Parameters:
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    
    Returns:
    str: The file info message.
    list: The options for the time column dropdown.
    list: The options for the value column dropdown.
    """
    df = parse_contents(contents, filename)
    if df.empty:
        return '❌ Failed to load file.', [], []
    opts = [{'label':c,'value':c} for c in df.columns]
    return f'✅ Loaded: {filename}', opts, opts

# Compute derivative and trends
@dash.callback(
    Output('ta-graph','figure'),
    Output('ta-summary','children'),
    Output('ta-computation-time','children'),
    Output('ta-alert-box','children'),
    Input('ta-btn','n_clicks'),
    State('ta-upload','contents'),
    State('ta-upload','filename'),
    State('ta-time-col','value'),
    State('ta-value-col','value'),
    State('ta-method','value'),
    State('ta-step','value'),
    State('ta-threshold','value'),
    prevent_initial_call=True
)
def compute_trends(n, contents, filename, tcol, vcol, method, step, thresh):
    """
    Compute the trends and update the graph and summary statistics.
    
    Parameters:
    n (int): The number of clicks on the compute button.
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    tcol (str): The name of the time column.
    vcol (str): The name of the value column.
    method (str): The method for computing the derivative.
    step (int): The step size for computing the derivative.
    thresh (float): The trend threshold.
    
    Returns:
    go.Figure: The updated graph.
    str: The updated summary statistics.
    str: The computation time.
    str: The alert message.
    """
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
    t = df[tcol]
    y = df[vcol].astype(float).values
    # time in seconds
    secs = (t - t.iloc[0]).dt.total_seconds().values
    # derivative array
    d = np.zeros_like(y)
    h = step
    if method=='forward':
        d[:-h] = (y[h:] - y[:-h]) / (secs[h:] - secs[:-h])
        d[-h:] = d[-h-1]
    elif method=='backward':
        d[h:] = (y[h:] - y[:-h]) / (secs[h:] - secs[:-h])
        d[:h] = d[h]
    else:  # central
        d[h:-h] = (y[2*h:] - y[:-2*h]) / (secs[2*h:] - secs[:-2*h])
        d[:h] = d[h]
        d[-h:] = d[-h-1]
    # classify trends
    trend = np.where(d>thresh, 'up', np.where(d<-thresh, 'down', 'flat'))
    # summary stats
    dt = np.diff(secs)
    up_time = np.sum(dt[trend[:-1]=='up'])/3600
    down_time = np.sum(dt[trend[:-1]=='down'])/3600
    avg_up = np.mean(d[trend=='up']) if np.any(trend=='up') else 0
    avg_down = np.mean(d[trend=='down']) if np.any(trend=='down') else 0
    summary = f"Up-trend: {up_time:.2f}h (avg slope {avg_up:.2f}), Down-trend: {down_time:.2f}h (avg slope {avg_down:.2f})"
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Value'))
    fig.add_trace(go.Scatter(x=t, y=d, mode='lines', name='Derivative', yaxis='y2'))
    # add colored segments
    colors = {'up':'green','down':'red','flat':'gray'}
    start=0
    curr=trend[0]
    for i in range(1,len(trend)):
        if trend[i]!=curr or i==len(trend)-1:
            seg_t = t.iloc[start:i+1]
            seg_y = y[start:i+1]
            fig.add_trace(go.Scatter(x=seg_t, y=seg_y, mode='lines', line={'color':colors[curr]}, showlegend=False))
            start=i
            curr=trend[i]
    # layout with secondary y-axis
    fig.update_layout(
        title='Trend Analysis',
        xaxis_title='Time',
        yaxis_title=vcol,
        yaxis2=dict(title='Derivative', overlaying='y', side='right'),
        plot_bgcolor='white',
        template='plotly_white'
    )
    computation_time = time.perf_counter() - start_time
    return fig, summary, f"Computed in {computation_time:.4f} seconds", create_alert(f"Success! Computed in {computation_time:.4f}s", 'success')
