# Frond-End file

# required libraries
import pandas as pd
import numpy as np

# dash library
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go


# defining main app layout

# start tha main app
# Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like.
# The second part describes the interactivity of the application
app = dash.Dash(__name__)


########################### test ##############################
my_layout = go.Layout(
    margin=dict(
        t=30,  # top margin: 30px, you want to leave around 30 pixels to
        # display the modebar above the graph.
        b=10,  # bottom margin: 10px
        #l=100,  # left margin: 10px
        #r=10,  # right margin: 10px
    ),
    width=900, height = 150
    # Some more layout options
)
################################################################

# define the layout
app.layout = html.Div(children=[

    html.H1(children="First layout of CoScorer", style={
            "textAlign": "center", "color": "#8FDBFF"}),

    html.Div(children=[
        html.Button(children="Load", style={"margin": "5px"}),
        html.Button(children="Save", style={"margin": "5px"}),
        html.Button(children="Run", style={"margin": "5px"}),
        html.Button(children="Empty", style={"margin": "5px"})
    ]),


    html.Spacer("--"),

    html.Div(children=[
        html.Div(children=[
            dcc.Graph(id="empty for th moment1", figure={
                      "layout": my_layout}),
            dcc.Graph(id="empty for th moment2", figure={
                      "layout": my_layout}),
            dcc.Graph(id="empty for th moment3", figure={
                      "layout": my_layout})
        ])
    ])

])


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
