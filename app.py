import dash
from dash import html, dcc

external_stylesheets = ['https://fonts.googleapis.com/css2?family=Roboto&display=swap']
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder='pages',
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)
server = app.server

navbar = html.Div(
    style={
        'display':'flex',
        'alignItems':'center',
        'justifyContent':'space-between',
        'padding':'10px',
        'background': 'linear-gradient(90deg, #f5f7fa, #c3cfe2)',
        'color':'#2c3e50',
        'position':'fixed',
        'top':'0',
        'width':'100%',
        'zIndex':'1000'
    },
    children=[
        html.Img(src='/assets/logo.png', style={'height':'40px','marginRight':'20px'}),
        html.Div(
            style={'display':'flex', 'justifyContent':'center', 'flexGrow':'1'},
            children=[
                dcc.Link('Floating-Point Demo', href='/floating-point-demo', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
                dcc.Link('Interpolation', href='/interpolation', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
                dcc.Link('Models', href='/models', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
                dcc.Link('Cross Sensitivity', href='/cross-sensitivity', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
                dcc.Link('Weighted Average', href='/optimization', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
                dcc.Link('Tools', href='/threshold-crossing', style={'marginRight':'15px','color':'#2c3e50','textDecoration':'none','fontWeight':'bold'}),
            ]
        )
    ])

app.layout = html.Div(
    [
        navbar,
        html.Div(dash.page_container, style={'paddingTop':'80px'})
    ],
    style={
        'fontFamily':'Roboto, sans-serif'
    }
)

if __name__ == '__main__':
    app.run(debug=True)
