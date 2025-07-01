import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception, parse_weights

# Register this page with Dash
register_page(__name__, path='/optimization', name='Weighted Average')

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
    # Page title
    html.H2('Weighted Sensor Average', style={'textAlign':'center','marginBottom':'30px'}),

    # File upload section
    html.Div([
        html.H4('Upload Sensor Data'),
        dcc.Upload(
            id='wa-upload',
            children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel File')]),
            style={
                'width':'100%','height':'60px','lineHeight':'60px',
                'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                'textAlign':'center','backgroundColor':'#fafafa'
            }, multiple=False
        ),
        html.Div(id='wa-file-info', style={'marginTop':'15px'})
    ], style=card_style),

    # Column selection
    html.Div([
        html.Div([html.Label('Time Column:'), dcc.Dropdown(id='wa-time-col')], style={'flex':1,'marginRight':'10px'}),
        html.Div([html.Label('Sensor Columns:'), dcc.Dropdown(id='wa-sensor-cols', multi=True)], style={'flex':1})
    ], style={**card_style,'display':'flex'}),

    # Weight configuration
    html.Div([
        html.Label('Weights (comma-separated):'),
        dcc.Input(id='wa-weights', type='text', placeholder='e.g., 0.3,0.3,0.4', style={'width':'100%'})
    ], style=card_style),

    # Compute button
    html.Div(html.Button('Compute Average', id='wa-btn', style=button_style), style={'textAlign':'center','marginBottom':'40px'}),

    # Results display
    dcc.Loading(children=[
        dcc.Graph(id='wa-graph'),
        html.Button('Download Averaged Series', id='wa-download-btn', style=button_style),
        dcc.Download(id='wa-download')
    ], type='default'),
    html.Div(id='wa-output', style={'marginTop':'20px'}),
    html.Div(id='wa-computation-time', style={'marginTop':'10px'}),
    html.Div(id='wa-alert-box', style={'marginTop':'20px'})
], style={
    'marginLeft':'240px','padding':'20px','paddingTop':'100px','fontFamily':'Roboto, sans-serif'
})

# helper to parse file
def parse_file(contents, filename):
    """
    Parse the uploaded file into a Pandas DataFrame.
    
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
    Output('wa-file-info','children'),
    Output('wa-time-col','options'),
    Output('wa-sensor-cols','options'),
    Input('wa-upload','contents'),
    State('wa-upload','filename'),
    prevent_initial_call=True
)
def update_wa_dropdowns(contents, filename):
    """
    Update the dropdown options based on the uploaded file.
    
    Parameters:
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    
    Returns:
    str: The file info message.
    list: The time column options.
    list: The sensor column options.
    """
    df = parse_file(contents, filename)
    if df.empty:
        return '❌ Failed to load file.', [], []
    opts = [{'label':c,'value':c} for c in df.columns]
    return f'✅ Loaded: {filename}', opts, opts

# Update callback with error handling
@dash.callback(
    Output('wa-graph','figure'),
    Output('wa-output','children'),
    Output('wa-computation-time','children'),
    Output('wa-alert-box','children'),
    Input('wa-btn','n_clicks'),
    [State('wa-upload','contents'),
     State('wa-upload','filename'),
     State('wa-time-col','value'),
     State('wa-sensor-cols','value'),
     State('wa-weights','value')],
    prevent_initial_call=True
)
def compute_weighted_average(n, contents, filename, tcol, sensors, weights_str):
    """
    Compute the weighted average of the sensor data.
    
    Parameters:
    n (int): The number of clicks on the compute button.
    contents (str): The contents of the uploaded file.
    filename (str): The name of the uploaded file.
    tcol (str): The time column name.
    sensors (list): The sensor column names.
    weights_str (str): The weights string.
    
    Returns:
    go.Figure: The weighted average graph.
    str: The weighted average output.
    str: The computation time.
    str: The alert message.
    """
    if n is None:
        raise dash.exceptions.PreventUpdate
    
    # Parse and validate
    df = parse_file(contents, filename)
    
    # Validate inputs
    errors = validate_inputs(
        df,
        required_columns=[tcol] + sensors,
        datetime_columns=[tcol],
        numeric_columns=sensors
    )
    
    if errors:
        return go.Figure(), "", "", create_alert("Validation errors: " + "; ".join(errors), 'error')
    
    # Parse weights
    weights, weight_error = parse_weights(weights_str)
    if weight_error:
        return go.Figure(), "", "", create_alert(weight_error, 'error')
    
    # Timing
    start_time = time.perf_counter()
    
    # ... existing computation ...
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)
    X = df[sensors].astype(float).values
    weights = np.array(weights)
    weights = weights / weights.sum()
    avg = X.dot(weights)
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[tcol], y=avg, mode='lines', name='Weighted Avg'))
    fig.update_layout(title='Weighted Average Sensor', xaxis_title='Time', yaxis_title='Value', plot_bgcolor='white', template='plotly_white')
    
    computation_time = time.perf_counter() - start_time
    
    return (
        fig,
        f"Weighted average: {avg.mean():.4f}",
        f"Computed in {computation_time:.4f} seconds",
        create_alert(f"Success! Computed in {computation_time:.4f}s", 'success')
    )

# download callback
@dash.callback(
    Output('wa-download','data'),
    Input('wa-download-btn','n_clicks'),
    State('wa-graph','figure'),
    prevent_initial_call=True
)
def download_wa(n, fig):
    """
    Download the weighted average data.
    
    Parameters:
    n (int): The number of clicks on the download button.
    fig (go.Figure): The weighted average graph.
    
    Returns:
    dcc.send_data_frame: The downloaded data.
    """
    y = fig['data'][0]['y']
    x = fig['data'][0]['x']
    df_out = pd.DataFrame({'time': x, 'weighted_avg': y})
    return dcc.send_data_frame(df_out.to_csv, 'weighted_average.csv', index=False)
