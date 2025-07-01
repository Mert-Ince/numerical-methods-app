import dash
from dash import html, dcc, Input, Output, State, register_page
import pandas as pd
import numpy as np
import base64, io
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

register_page(__name__, path='/ode-solver', name='ODE Solver')

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
    html.Div([
        html.H4('Tools', style={'marginBottom':'20px'}),
        dcc.Link('Threshold Crossing', href='/threshold-crossing', style=link_style),
        dcc.Link('Exposure Timing', href='/exposure-timing', style=link_style),
        dcc.Link('Trend Analysis', href='/trend-analysis', style=link_style),
        dcc.Link('Integration', href='/integration', style=link_style),
        dcc.Link('ODE Solver', href='/ode-solver', style=link_style)
    ], style=sidebar_style),

    html.Div([
        html.H2('ODE Solver: First-order Decay with Emissions', style={'textAlign':'center','marginBottom':'30px'}),

        html.Div([
            html.H4('Upload Emission Data (optional)'),
            dcc.Upload(
                id='ode-upload',
                children=html.Div(['📂 Drag & Drop or ', html.A('Select CSV/Excel file')]),
                style={
                    'width':'100%','height':'60px','lineHeight':'60px',
                    'borderWidth':'2px','borderStyle':'dashed','borderRadius':'8px',
                    'textAlign':'center','backgroundColor':'#fafafa'
                }, multiple=False
            ),
            html.Div(id='ode-file-info', style={'marginTop':'15px'})
        ], style=card_style),

        html.Div([  # Parameters
            html.Div([html.Label('Loss rate k (1/time):'), dcc.Input(id='ode-k', type='number', value=0.1, step=0.01)], style={'flex':1,'marginRight':'10px'}),
            html.Div([html.Label('Initial C0:'), dcc.Input(id='ode-c0', type='number', value=0.0, step=0.1)], style={'flex':1}),
        ], style={**card_style,'display':'flex'}),
        html.Div([  # Solver options
            html.Div([html.Label('Solver Method:'), dcc.Dropdown(
                id='ode-method',
                options=[{'label':'RK45','value':'RK45'},{'label':'RK23','value':'RK23'},{'label':'BDF','value':'BDF'}],
                value='RK45'
            )], style={'flex':1,'marginRight':'10px'}),
            html.Div([html.Label('Tolerance:'), dcc.Input(id='ode-tol', type='number', value=1e-6, step=1e-7)], style={'flex':1})
        ], style={**card_style,'display':'flex'}),

        html.Div(html.Button('Solve ODE', id='ode-btn', style=button_style), style={'textAlign':'center','marginBottom':'40px'}),

        dcc.Loading(children=[dcc.Graph(id='ode-graph')], type='default'),

        html.Div(id='ode-output', style={'marginTop':'30px','textAlign':'center','fontWeight':'bold'})
    ], style={
        'marginLeft':'240px','padding':'20px','paddingTop':'100px','fontFamily':'Roboto, sans-serif'
    })
])

# parse helper
def parse_file(contents, filename):
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

# callbacks
@dash.callback(
    Output('ode-file-info','children'),
    Output('ode-upload','contents','filename'),
    Input('ode-upload','contents'),
    State('ode-upload','filename'),
    prevent_initial_call=True
)
def update_ode_file(contents, filename):
    if contents:
        return f'✅ Loaded: {filename}', contents, filename
    return '❌ Failed to load file.', None, None

@dash.callback(
    Output('ode-graph','figure'),
    Output('ode-output','children'),
    Input('ode-btn','n_clicks'),
    State('ode-upload','contents'),
    State('ode-upload','filename'),
    State('ode-k','value'),
    State('ode-c0','value'),
    State('ode-method','value'),
    State('ode-tol','value'),
    prevent_initial_call=True
)
def solve_ode(n, contents, filename, k, c0, method, tol):
    # prepare time span and emission function
    if contents:
        df = parse_file(contents, filename)
        df['time'] = pd.to_datetime(df.iloc[:,0])
        t = (df['time'] - df['time'].iloc[0]).dt.total_seconds().values
        e = df.iloc[:,1].astype(float).values
        def E_fun(ti):
            return np.interp(ti, t, e)
        t_span = (t[0], t[-1])
        t_eval = t
    else:
        t_span = (0, 3600*24)
        t_eval = np.linspace(*t_span, 200)
        E_fun = lambda ti: 0

    # ODE: dC/dt = E(t) - k*C
    def f(ti, Ci):
        return E_fun(ti) - k*Ci

    sol = solve_ivp(f, t_span, [c0], method=method, atol=tol, rtol=tol, t_eval=t_eval)
    times = sol.t
    C = sol.y[0]

    # convert time back to datetime
    if contents:
        start = df['time'].iloc[0]
        times_dt = start + pd.to_timedelta(times, unit='s')
    else:
        times_dt = times

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times_dt, y=C, mode='lines', name='C(t)'))
    fig.update_layout(title='ODE Solution', xaxis_title='Time', yaxis_title='Concentration', plot_bgcolor='white', template='plotly_white')
    stats = f"Solver '{method}' steps: {sol.nfev}, success: {sol.success}"
    return fig, stats
