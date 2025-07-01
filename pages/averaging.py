import dash
from dash import html, dcc, Input, Output, State, register_page
import numpy as np
import plotly.graph_objects as go

# Register this page with Dash
register_page(__name__, path='/floating-point-demo', name='Floating-Point Demo')

# Constants
N_SAMPLES = 1440  # Number of samples for floating-point averaging demo
ROUND_DECIMALS = 2  # Number of decimal places to round to in rounding error demo

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
    html.H2('Floating-Point Averaging Error Demo', style={'textAlign': 'center', 'marginBottom': '30px'}),

    # Floating-point averaging demo
    html.Div([
        # Input field for value to average
        html.Label('Enter a value to average:'),
        dcc.Input(
            id='fp-input',
            type='number',
            value=0.333,
            step=0.0001,
            style={'width': '120px', 'marginRight': '10px'}
        ),
        # Compute button
        html.Button('Compute Averages', id='fp-btn', style=button_style),
        # Output field for results
        html.Div(id='fp-output', style={'marginTop': '15px'})
    ], style=card_style),

    # Rounding error demo
    html.Div([
        # Title for rounding error demo
        html.H4('Rounding Error Demo', style={'marginBottom': '10px'}),
        # Input for number of samples
        html.Label('Number of Samples (N_SAMPLES):'),
        dcc.Input(
            id='round-samples-input',
            type='number',
            value=N_SAMPLES,
            min=1,
            step=1,
            style={'width': '120px', 'marginRight': '10px'}
        ),
        # Input for decimal places to round
        html.Label('Decimal Places to Round (ROUND_DECIMALS):'),
        dcc.Input(
            id='round-decimals-input',
            type='number',
            value=ROUND_DECIMALS,
            min=0,
            step=1,
            style={'width': '120px', 'marginRight': '10px'}
        ),
        # Compute button
        html.Button('Compute Rounding Demo', id='round-btn', style=button_style),
        # Output field for results
        html.Div(id='round-output', style={'marginTop': '15px'})
    ], style=card_style)
], style={
    'maxWidth': '600px',
    'margin': 'auto',
    'padding': '20px',
    'fontFamily': 'Roboto, sans-serif',
    'backgroundColor': '#f5f5f5'
})

# Callback function for floating-point averaging demo
@dash.callback(
    Output('fp-output', 'children'),
    Input('fp-btn', 'n_clicks'),
    State('fp-input', 'value'),
    prevent_initial_call=True
)
def compute_fp_error(n_clicks, val):
    """
    Compute floating-point averaging error.

    Parameters:
    n_clicks (int): Number of times the compute button has been clicked.
    val (float): Value to average.

    Returns:
    list: List of HTML elements displaying the results.
    """
    # Create arrays of float32 and float64 values
    data32 = np.full(N_SAMPLES, val, dtype=np.float32)
    data64 = np.full(N_SAMPLES, val, dtype=np.float64)
    # Compute sum and average of values
    total32, total64 = data32.sum(), data64.sum()
    avg32, avg64 = total32 / N_SAMPLES, total64 / N_SAMPLES
    # Compute error in average
    error32, error64 = avg32 - val, avg64 - val

    # Create figure for results
    fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])])
    fig.update_layout(template='plotly_white')

    # Return list of HTML elements displaying results
    return html.Div([
        html.P('Using float32:'),
        html.Ul([
            html.Li(f'Sum     = {total32:.15f}'),
            html.Li(f'Average = {avg32:.15f}'),
            html.Li(f'Error   = {error32:.15e}')
        ]),
        html.P('Using float64:'),
        html.Ul([
            html.Li(f'Sum     = {total64:.15f}'),
            html.Li(f'Average = {avg64:.15f}'),
            html.Li(f'Error   = {error64:.15e}')
        ])
    ])

# Callback function for rounding error demo
@dash.callback(
    Output('round-output', 'children'),
    Input('round-btn', 'n_clicks'),
    State('fp-input', 'value'),
    State('round-samples-input', 'value'),
    State('round-decimals-input', 'value'),
    prevent_initial_call=True
)
def compute_round_error(n_clicks, val, n_samples, round_decimals):
    """
    Compute rounding error.

    Parameters:
    n_clicks (int): Number of times the compute button has been clicked.
    val (float): Value to average.
    n_samples (int): Number of samples.
    round_decimals (int): Number of decimal places to round to.

    Returns:
    list: List of HTML elements displaying the results.
    """
    # Compute ideal sum
    ideal = val * n_samples
    # Initialize total to zero
    total = 0.0
    # Round each addition to ROUND_DECIMALS decimal places
    for _ in range(n_samples):
        total = round(total + val, round_decimals)
    # Compute average
    avg = total / n_samples
    # Compute sum and average error
    sum_err = total - ideal
    avg_err = avg - val

    # Return list of HTML elements displaying results
    return html.Div([
        html.P(f'Rounding to {round_decimals} decimal places each step:'),
        html.Ul([
            html.Li(f'Ideal Sum      = {ideal:.{round_decimals}f}'),
            html.Li(f'Rounded Sum    = {total:.{round_decimals}f}'),
            html.Li(f'Sum Error      = {sum_err:.{round_decimals}f}')
        ]),
        html.Ul([
            html.Li(f'Ideal Average  = {val:.{round_decimals}f}'),
            html.Li(f'Rounded Average= {avg:.15f}'),
            html.Li(f'Average Error  = {avg_err:.15e}')
        ])
    ])
