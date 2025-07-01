import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io, time
import plotly.graph_objects as go
from scipy.linalg import lu_factor, lu_solve

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

# Register this page with Dash
register_page(__name__, path='/cross-sensitivity', name='Cross Sensitivity')

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
    html.Div([
        html.H2('Cross-Sensitivity Calibration', style={'textAlign':'center','marginBottom':'30px'}),
        
        # Upload calibration matrix A
        html.Div([
            html.H4('Upload Calibration Matrix A'),
            dcc.Upload(
                id='cs-upload-matrix',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select Matrix CSV')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='cs-matrix-info', style={'marginTop':'15px'})
        ], style=card_style),
        
        # Upload sensor readings b(t)
        html.Div([
            html.H4('Upload Sensor Readings b(t)'),
            dcc.Upload(
                id='cs-upload-b',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select Readings CSV')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='cs-b-info', style={'marginTop':'15px'})
        ], style=card_style),
        
        # Column selection
        html.Div([
            html.Div([html.Label('Time Column:'), dcc.Dropdown(id='cs-time-col')], style={'flex':1,'marginRight':'10px'}),
            html.Div([html.Label('Sensor Columns:'), dcc.Dropdown(id='cs-b-cols', multi=True)], style={'flex':1})
        ], style={**card_style,'display':'flex'}),
        
        # Compute button
        html.Div(html.Button('Compute Calibration', id='cs-btn', style=button_style),
                 style={'textAlign':'center','marginBottom':'40px'}),
        
        # Results display
        dcc.Loading(children=[dcc.Graph(id='cs-graph')], type='default'),
        html.Div(id='cs-output', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'}),
        html.Div(id='cs-computation-time', style={'marginTop':'10px','textAlign':'center'}),
        html.Div(id='cs-alert-box', style={'marginTop':'10px','textAlign':'center'})
    ], style={
        'marginLeft':'240px','padding':'20px','paddingTop':'100px',
        'fontFamily':'Roboto, sans-serif'
    }),
    
    # Data stores
    dcc.Store(id='cs-matrix-store'),
    dcc.Store(id='cs-b-store')
])

# Helper function to parse uploaded file contents
def parse_contents(contents, filename):
    # Split the contents into two parts: the content type and the content string
    ct, cs = contents.split(',')
    # Decode the content string
    decoded = base64.b64decode(cs)
    try:
        # Try to read the file as a CSV or Excel file
        if filename.lower().endswith('.csv'):
            return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.lower().endswith(('.xls','.xlsx')):
            return pd.read_excel(io.BytesIO(decoded))
    except Exception:
        # If the file cannot be read, return an empty DataFrame
        pass
    return pd.DataFrame()

# Populate matrix and b stores
@dash.callback(
    Output('cs-matrix-info','children'),
    Output('cs-matrix-store','data'),
    Input('cs-upload-matrix','contents'),
    State('cs-upload-matrix','filename'),
    prevent_initial_call=True
)
def load_matrix(contents, filename):
    # Parse the uploaded file contents
    df = parse_contents(contents, filename)
    if df.empty:
        # If the file is empty, return an error message
        return '❌ Failed to load matrix.', None
    try:
        # Try to convert the DataFrame to a NumPy array
        A = df.values
        n,_ = A.shape
        # Return the matrix information and store the matrix data
        return f'✅ Loaded matrix ({n}×{n})', A.tolist()
    except:
        # If the matrix is invalid, return an error message
        return '❌ Invalid matrix.', None

@dash.callback(
    Output('cs-b-info','children'),
    Output('cs-time-col','options'),
    Output('cs-b-cols','options'),
    Output('cs-b-store','data'),
    Input('cs-upload-b','contents'),
    State('cs-upload-b','filename'),
    prevent_initial_call=True
)
def load_b(contents, filename):
    # Parse the uploaded file contents
    df = parse_contents(contents, filename)
    if df.empty:
        # If the file is empty, return an error message
        return '❌ Failed to load readings.', [], [], None
    cols = [{'label':c,'value':c} for c in df.columns]
    # Return the readings information and store the readings data
    return f'✅ Loaded readings ({len(df)} rows)', cols, cols, df.to_dict('records')

# Compute calibration
@dash.callback(
    Output('cs-graph','figure'),
    Output('cs-output','children'),
    Output('cs-computation-time','children'),
    Output('cs-alert-box','children'),
    Input('cs-btn','n_clicks'),
    [State('cs-matrix-store','data'),
     State('cs-b-store','data'),
     State('cs-time-col','value'),
     State('cs-b-cols','value')],
    prevent_initial_call=True
)
def compute_calibration(n, matrix_data, b_data, tcol, bcols):
    if n is None:
        # If the button has not been clicked, prevent the callback from running
        raise dash.exceptions.PreventUpdate
    
    # Validate inputs
    if matrix_data is None:
        # If the matrix data is missing, return an error message
        return go.Figure(), "", "", create_alert("Matrix data is missing", 'error')
    if b_data is None:
        # If the readings data is missing, return an error message
        return go.Figure(), "", "", create_alert("Sensor readings data is missing", 'error')
    if not tcol:
        # If the time column is not selected, return an error message
        return go.Figure(), "", "", create_alert("Time column not selected", 'error')
    if not bcols:
        # If the sensor columns are not selected, return an error message
        return go.Figure(), "", "", create_alert("Sensor columns not selected", 'error')
    
    # Timing
    start_time = time.perf_counter()
    
    # Convert the matrix data to a NumPy array
    A = np.array(matrix_data)
    # Convert the readings data to a Pandas DataFrame
    dfb = pd.DataFrame(b_data)
    # Convert the time column to datetime format
    dfb[tcol] = pd.to_datetime(dfb[tcol])
    # Sort the readings by time
    dfb = dfb.sort_values(tcol).reset_index(drop=True)
    # Get the time values
    times = dfb[tcol]
    # Get the sensor readings
    B = dfb[bcols].astype(float).values
    
    # Direct solve
    start = time.perf_counter()
    # Solve the system of linear equations using NumPy's solve function
    X_direct = np.linalg.solve(A, B.T).T
    # Get the time taken for the direct solve
    t_direct = time.perf_counter() - start
    
    # LU solve
    start = time.perf_counter()
    # Perform LU decomposition on the matrix A
    lu, piv = lu_factor(A)
    # Solve the system of linear equations using the LU decomposition
    X_lu = lu_solve((lu,piv), B.T).T
    # Get the time taken for the LU solve
    t_lu = time.perf_counter() - start
    
    # Plot the first two species if they exist
    fig = go.Figure()
    for i,col in enumerate(bcols):
        # Add a scatter plot for each species
        fig.add_trace(go.Scatter(x=times, y=X_lu[:,i], mode='lines', name=f'Conc {col}'))
    # Update the plot layout
    fig.update_layout(title='Estimated Concentrations', xaxis_title='Time', yaxis_title='Concentration', plot_bgcolor='white', template='plotly_white')
    # Get the output string
    out = f"Direct solve: {t_direct:.4f}s, LU solve: {t_lu:.4f}s (speedup {t_direct/t_lu:.1f}×)"
    
    # Get the computation time
    computation_time = time.perf_counter() - start_time
    
    # Return the plot, output string, computation time, and success alert
    return (
        fig,
        out,
        f"Computed in {computation_time:.4f} seconds",
        create_alert(f"Success! Computed in {computation_time:.4f}s", 'success')
    )
