import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import base64, io, time
import plotly.graph_objects as go

# Import shared utilities
from utils import validate_inputs, create_alert, global_catch_exception

# Register this page with Dash
register_page(__name__, path='/models', name='Models')

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

# Main page layout
layout = html.Div([
    # Page title
    html.H2('Linear Regression Model', style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    # File upload section for feature data
    html.Div([
        html.H4('Upload Feature Data'),
        dcc.Upload(
            id='models-upload-features',
            children=html.Div(['📁 Drag & Drop or ', html.A('Select Features File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                'textAlign': 'center', 'marginBottom': '10px', 'backgroundColor': '#fafafa'
            }, multiple=False
        ),
        html.Div(id='models-file-info-features')
    ], style=card_style),
    
    # File upload section for reference data
    html.Div([
        html.H4('Upload Reference Data'),
        dcc.Upload(
            id='models-upload-reference',
            children=html.Div(['📁 Drag & Drop or ', html.A('Select Reference File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '2px', 'borderStyle': 'dashed', 'borderRadius': '8px',
                'textAlign': 'center', 'marginBottom': '10px', 'backgroundColor': '#fafafa'
            }, multiple=False
        ),
        html.Div(id='models-file-info-reference')
    ], style=card_style),
    
    # Data alignment and model configuration
    html.Div([
        html.Div([html.Label('Feature Time Column:'), dcc.Dropdown(id='models-time-col-features')], style={'flex':1, 'marginRight':'10px'}),
        html.Div([html.Label('Reference Time Column:'), dcc.Dropdown(id='models-time-col-reference')], style={'flex':1})
    ], style={**card_style, 'display':'flex'}),
    
    html.Div([
        html.Div([html.Label('Covariates (Features):'), dcc.Dropdown(id='models-feature-cols', multi=True)], style={'flex':1, 'marginRight':'10px'}),
        html.Div([html.Label('Reference Column:'), dcc.Dropdown(id='models-target-col')], style={'flex':1})
    ], style={**card_style, 'display':'flex'}),
    
    html.Div([
        html.Div([html.Label('Resample Frequency:'), dcc.Input(id='models-resample-freq', type='text', value='H')], style={'flex':1, 'marginRight':'10px'}),
        html.Div([html.Label('Offset (minutes):'), dcc.Input(id='models-resample-offset', type='number', value=60)], style={'flex':1})
    ], style={**card_style, 'display':'flex'}),
    
    html.Div(html.Button('Train Model', id='models-train-btn', style=button_style), style={'textAlign':'center', 'marginBottom':'20px'}),
    
    # Model coefficients display
    html.Div(id='model-coefs', style={'textAlign':'center', 'marginBottom':'20px', 'fontWeight':'bold'}),
    
    # Results graph
    dcc.Loading(dcc.Graph(id='model-graph'), type='default'),
    
    # Alert box for user feedback
    html.Div(id='alert-box', style={'textAlign':'center', 'marginBottom':'20px'}),
    
    # Computation time display
    html.Div(id='computation-time', style={'textAlign':'center', 'marginBottom':'20px'})
], style={'maxWidth':'900px', 'margin':'auto', 'padding':'20px', 'fontFamily':'Roboto, sans-serif', 'backgroundColor':'#f5f5f5'})


def parse_file(contents, filename):
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
    Output('models-file-info-features', 'children'),
    Output('models-time-col-features', 'options'),
    Output('models-feature-cols', 'options'),
    Output('models-features-store', 'data'),
    Input('models-upload-features', 'contents'),
    State('models-upload-features', 'filename'),
    prevent_initial_call=True
)
def update_features(contents, filename):
    df = parse_file(contents, filename)
    if df.empty:
        return '❌ Failed to load features file.', [], [], None
    options = [{'label': col, 'value': col} for col in df.columns]
    return f'✅ Loaded: {filename}', options, options, df.to_dict('records')

@dash.callback(
    Output('models-file-info-reference', 'children'),
    Output('models-time-col-reference', 'options'),
    Output('models-target-col', 'options'),
    Output('models-reference-store', 'data'),
    Input('models-upload-reference', 'contents'),
    State('models-upload-reference', 'filename'),
    prevent_initial_call=True
)
def update_reference(contents, filename):
    df = parse_file(contents, filename)
    if df.empty:
        return '❌ Failed to load reference file.', [], [], None
    options = [{'label': col, 'value': col} for col in df.columns]
    return f'✅ Loaded: {filename}', options, options, df.to_dict('records')

@dash.callback(
    Output('model-coefs', 'children'),
    Output('model-graph', 'figure'),
    Output('alert-box', 'children'),
    Output('computation-time', 'children'),
    Input('models-train-btn', 'n_clicks'),
    State('models-features-store', 'data'),
    State('models-reference-store', 'data'),
    State('models-time-col-features', 'value'),
    State('models-time-col-reference', 'value'),
    State('models-feature-cols', 'value'),
    State('models-target-col', 'value'),
    State('models-resample-freq', 'value'),
    State('models-resample-offset', 'value'),
    prevent_initial_call=True
)
def train_and_plot(n_clicks, features_data, reference_data, time_feat, time_ref, features, target, freq, offset):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    
    # Parse and validate
    df_feat = pd.DataFrame(features_data)
    df_ref = pd.DataFrame(reference_data)
    
    # Validate inputs
    errors = validate_inputs(
        df_feat,
        required_columns=[time_feat] + features,
        datetime_columns=[time_feat],
        numeric_columns=features
    )
    errors += validate_inputs(
        df_ref,
        required_columns=[time_ref, target],
        datetime_columns=[time_ref],
        numeric_columns=[target]
    )
    
    if errors:
        return "", go.Figure(), create_alert("Validation errors: " + "; ".join(errors), 'error'), ""
    
    # Timing
    start_time = time.perf_counter()
    
    # ... existing computation ...
    df_feat[time_feat] = pd.to_datetime(df_feat[time_feat])
    df_ref[time_ref] = pd.to_datetime(df_ref[time_ref])
    df_feat = df_feat.set_index(time_feat)
    df_feat = df_feat.resample(freq, offset=pd.Timedelta(minutes=offset)).mean().reset_index()
    t_min, t_max = df_feat[time_feat].min(), df_feat[time_feat].max()
    df_ref = df_ref[(df_ref[time_ref] >= t_min) & (df_ref[time_ref] <= t_max)]
    merged = pd.merge(df_feat, df_ref, left_on=time_feat, right_on=time_ref, how='inner')
    merged = merged.sort_values(time_feat)
    X = merged[features].astype(float).values
    y = merged[target].astype(float).values
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    coef_text = ' | '.join([f'{feat}: {coef:.4f}' for feat, coef in zip(features, model.coef_)])
    intercept_text = f'Intercept: {model.intercept_:.4f}'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged[time_feat], y=merged[target], mode='markers+lines', name='Reference'))
    for feat in features:
        fig.add_trace(go.Scatter(x=merged[time_feat], y=merged[feat], mode='lines', name=f'Feature: {feat}'))
    fig.add_trace(go.Scatter(x=merged[time_feat], y=y_pred, mode='lines', name='Predicted'))
    fig.update_layout(
        title='Regression Results',
        xaxis_title='Time',
        yaxis_title='Value',
        plot_bgcolor='white',
        template='plotly_white'
    )
    computation_time = time.perf_counter() - start_time
    
    return html.Div([html.P(coef_text), html.P(intercept_text)]), fig, create_alert(f"Success! Computed in {computation_time:.4f}s", 'success'), f"Computed in {computation_time:.4f} seconds"