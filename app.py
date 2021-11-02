# Frond-End file

# required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# dash library
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc


# defining main app layout

# start tha main app
# Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like.
# The second part describes the interactivity of the application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# First row buttons
load_button = dbc.Button("Load")

save_button = dbc.Button("Save")

help_button = dbc.Button("Help")

edit_button = dbc.Button("Edit")

params_button = dbc.Button("Parameters")


# Abstract buttons
header_buttons = html.Div(children=[
    load_button,
    save_button,
    help_button,
    edit_button,
    params_button
], className="d-grid gap-1 d-md-flex")


# Graph div
trace_graphs = html.Div(children=[
    dcc.Graph(id="ch1"),
    dcc.Graph(id="ch2"),
    dcc.Graph(id="ch3")
])


# confusion matrix table
# creating df which mimics sklearn output
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
cm = confusion_matrix(y_true, y_pred, normalize='true')
class_names = ['Type1', 'Type2', 'Type3']
df = pd.DataFrame(np.round(cm, 3), columns=class_names, index=class_names)

table = dbc.Table.from_dataframe(
    df, striped=False, bordered=False, hover=True, index=True, responsive=True
)


# define app layout using dbc container
app.layout = dbc.Container(
    [html.H1("SleeZy v1.0.0"), header_buttons, trace_graphs, table], fluid=True)

# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)


"""

########################### test ##############################
my_layout = go.Layout(
    margin=dict(
        t=10,  # top margin: 30px, you want to leave around 30 pixels to
        # display the modebar above the graph.
        b=10,  # bottom margin: 10px
        # l=100,  # left margin: 10px
        # r=10,  # right margin: 10px
    ),
    width=900, height=150,
    autosize=True
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
    ], style={"display": "flex", "align-items": "center", "justify-content": "center"}),

    html.Div(children=[dcc.Graph(id="n1"), dcc.Graph(
        id="n2")], className="row")

])
"""
