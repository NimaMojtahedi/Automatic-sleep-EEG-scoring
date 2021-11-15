# Frontend file

# required libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb
import os
import subprocess

# internal files and functions
from utils import process_input_data

# dash library
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html
import plotly.express as px
from plotly.subplots import make_subplots

# dash extension
from dash_extensions import Keyboard

# dash uploader
import dash_uploader as du

# university logo path
University_Logo = "https://upload.wikimedia.org/wikipedia/de/9/97/Eberhard_Karls_Universit%C3%A4t_T%C3%BCbingen.svg"

# search bar
search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search epoch", color="primary", className="ms-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

# save button (deparcated)
#save_button = dbc.Button("Save", id="save-button", className="me-2", size="sm")

# navigation toolbar with logo, software title and a button
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=University_Logo, height="40px")),
                        dbc.Col(dbc.NavbarBrand("Sleeezy", className="ms-2")),
                        dbc.Button("About Us", id="about-us-button",
                                   className="me-2", size="sm")
                    ],
                    align="left",
                    className="g-0",
                ),
                href="http://www.physiologie2.uni-tuebingen.de/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="info",
    dark=True,
)

# draft training parameters list
input_params = html.Div([
    dbc.InputGroup([dbc.InputGroupText("ML algorithm"), dbc.Input(
        placeholder="Which ML you want to use?")], className="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("nThread"), dbc.Input(
        placeholder="The number of dedicated threads")], className="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("GPU"), dbc.Input(
        placeholder="1 for yes, 0 for no")], className="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("Lag time"), dbc.Input(
        placeholder="How much you can wait for training")], className="mb-1")
])

# config menu items
config_menu_items = html.Div([
    dbc.DropdownMenuItem("Human", id="dropdown-menu-item-1"),
    dbc.DropdownMenuItem("Rodent", id="dropdown-menu-item-2"),
    dbc.DropdownMenuItem(divider=True),
    dbc.DropdownMenuItem("Custom", id="dropdown-menu-item-3"),
])

# import configs
input_config = html.Div([
    dbc.InputGroup([
        dbc.DropdownMenu(config_menu_items, label="Species"),
        dbc.Input(id="input-group-dropdown-input", placeholder="Epoch length")]),
    dbc.InputGroup([dbc.InputGroupText("Channels to import"), dbc.Input(
        placeholder="How many channels do you have?")], className="mb-1")
])


# initialize header buttons
# help button
help_button = dbc.Button("Help", id="help-button", className="me-2", size="sm")
# edit buttons
edit_button = dbc.Button("upload", id="upload-button",
                         className="me-2", size="sm")

# import button
import_collapse = html.Div([
    dbc.Button("Import", id="import_collapse_button",
               className="me-2", n_clicks=0),
    dbc.Collapse(input_config, id="import_collapse", is_open=False)
])

# advanced parameters button
param_collapse = html.Div([
    dbc.Button("Advanced parameters", id="param_collapse_button",
               className="mb-3", n_clicks=0, size="sm"),
    dbc.Collapse(input_params, id="param_collapse", is_open=False)
])


# Abstract button
header_buttons = html.Div(dbc.Row([
    dbc.Col(
        html.Div(children=[import_collapse, edit_button, help_button]), width=4),
    dbc.Col(html.Div(children=param_collapse), width=4)
]))


def plot_traces(index):

    # call trace from inside
    trace = np.random.rand((1000)) + index  # still need to fix X axis (time)

    # get trace length
    trace_len = len(trace)

    # vertical ines positions
    x0 = trace_len/3
    x1 = (2 * trace_len) / 3

    y0 = trace.min()
    y1 = trace.max()

    y0 -= y1 * 0.2
    y1 += y1 * 0.2

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        print_grid=False, vertical_spacing=0.05)

    # changing px.line(y=trace)["data"][0] to go.Scatter(y=trace, mode="lines")
    # increase speed by factor of ~5
    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=3, col=1)

    # adding lines (alternative is box)
    split_line1 = go.Scatter(x=[x0, x0], y=[y0, y1], mode="lines",
                             hovertext="Epoch divider",
                             line=dict(color='red', width=6, dash='dash'))
    split_line2 = go.Scatter(x=[x1, x1], y=[y0, y1], mode="lines",
                             hovertext="Epoch divider",
                             line=dict(color='red', width=6, dash='dash'))

    fig.add_trace(split_line1, row=1, col=1)
    fig.add_trace(split_line2, row=1, col=1)

    fig.add_trace(split_line1, row=2, col=1)
    fig.add_trace(split_line2, row=2, col=1)

    fig.add_trace(split_line1, row=3, col=1)
    fig.add_trace(split_line2, row=3, col=1)

    fig.update_layout(margin=dict(l=100, r=100, t=1, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      width=900, height=400, showlegend=False
                      )
    return fig


# get accuracy plot
def get_acc_plot():

    ### call any function to receive train, val accuracy values inside this function #####

    # this will change with real data
    df = pd.DataFrame(
        {"Train": np.exp(-1/np.arange(2, 10, .1)), "Validation":  np.exp(-1/np.arange(2, 10, .1)) + .05})

    # start plotting
    fig = px.line(df, y=["Train", "Validation"])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=1, r=1, t=1, b=1),
                      legend=dict(
                          yanchor="bottom",
                          y=0.1,
                          xanchor="left",
                          x=0.45),
                      xaxis={"showgrid": False, "showline": True},
                      yaxis={"showgrid": False, "showline": True}
                      )

    return fig


# spectrum & histograms
def get_hists():

    # call trace from inside
    trace = np.random.rand((100))

    fig = make_subplots(rows=2, cols=3, shared_yaxes=True,
                        print_grid=False, vertical_spacing=0.25, horizontal_spacing=0.05,
                        subplot_titles=("Power Spectrums", "", "", "Amplitude Histograms", "", ""))

    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=1, col=1)
    fig.add_trace(go.Histogram(x=trace, marker_color='LightSkyBlue',
                  opacity=0.75, histnorm="probability"), row=2, col=1)
    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=1, col=2)
    fig.add_trace(go.Histogram(x=trace, marker_color='#330C73',
                  opacity=0.75, histnorm="probability"), row=2, col=2)
    fig.add_trace(go.Scatter(y=trace, mode="lines", line=dict(
        color='black', width=1)), row=1, col=3)
    fig.add_trace(go.Histogram(x=trace, marker_color='#330C73',
                  opacity=0.75, histnorm="probability"), row=2, col=3)

    # in case it is necessary for histograms xbins=dict(start=-3.0,end=4,size=0.5)

    fig.update_layout(margin=dict(l=1, r=1, t=20, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', bargap=0.25,
                      showlegend=False
                      )

    return fig


def get_confusion_mat():

    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    class_names = ['Type1', 'Type2', 'Type3']
    df = pd.DataFrame(np.round(cm, 3), columns=class_names, index=class_names)

    return df


# start tha main app
# Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like.
# The second part describes the interactivity of the application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY])


# Graph div
trace_graphs = html.Div(children=[
    dcc.Graph(id="ch", responsive=True),
])

# lower Row (contains all learning graphs and informations + spectrums and histograms)
# confusion matrix table
# creating df which mimics sklearn output
table = dbc.Table.from_dataframe(
    get_confusion_mat(), striped=False, bordered=False, hover=True, index=True, responsive=True, size="sm"
)

# accuracy graph
acc_graph = dcc.Graph(id="accuracy", figure=get_acc_plot(
), responsive=True, style={"width": "40vh", "height": "30vh"})


# hist graphs
hist_graph = dcc.Graph(id="hist-graphs", figure=get_hists(),
                       responsive=True, style={"width": "90vh", "height": "30vh"})


# lower row left-side
lower_row_left = dbc.Row([
    dbc.Col(table, width=3, align="center"),
    dbc.Col(acc_graph, width=5, align="center"),
    dbc.Col(dbc.Card("Train Info", id="train_info"), width=4)
], style={"display": "flex"})

# lower row right-side
lower_row_right = dbc.Row([
    hist_graph
])

# lower row
lower_row = html.Div(children=[
    html.H3("Back-End informations",
            style={"border": "2px solid powderblue", "margin-bottom": "0.67em", "margin-top": "1em"}),
    dbc.Row([
        dbc.Col(lower_row_left, width=7),
        dbc.Col(lower_row_right, width=5)
    ], className="g-5", align="end")
])


# detecting keyboard keys
my_keyboard = html.Div(Keyboard(id="keyboard"))


# define app layout using dbc container
app.layout = dbc.Container(
    html.Div([navbar, header_buttons, trace_graphs, lower_row, my_keyboard]), fluid=True)

# all callBacks


@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("import_collapse", "is_open"),
    [Input("import_collapse_button", "n_clicks")],
    [State("import_collapse", "is_open")],
)
def toggle_import_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output("input-group-dropdown-input", "value"),
    [
        Input("dropdown-menu-item-1", "n_clicks"),
        Input("dropdown-menu-item-2", "n_clicks"),
        Input("dropdown-menu-item-3", "n_clicks"),
    ],
)
def on_button_click(n1, n2, n3):
    ctx = dash.callback_context

    if not ctx.triggered:
        return ""
    else:
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "dropdown-menu-item-3":
        return "Enter your customized epoch length"
    elif button_id == "dropdown-menu-item-1":
        return "30"
    else:
        return "10"


@app.callback(
    Output("param_collapse", "is_open"),
    [Input("param_collapse_button", "n_clicks")],
    [State("param_collapse", "is_open")],
)
def toggle_param_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    [Output("ch", "figure"), Output("hist-graphs", "figure")],
    [Input("keyboard", "keydown"), Input("keyboard", "n_keydowns")]
)
def keydown(event, n_keydowns):
    print(n_keydowns)
    if n_keydowns:
        if event["key"] == "ArrowLeft":

            # traces plot
            fig_traces = plot_traces(index=np.random.randint(0, 10, 1))

            # accuracy plot
            fig_acc = get_hists()

        elif event["key"] == "ArrowRight":

            # traces plot
            fig_traces = plot_traces(index=np.random.randint(0, 10, 1))

            # accuracy plot
            fig_acc = get_hists()

        print(event["key"], n_keydowns)
        return fig_traces, fig_acc
    return plot_traces(index=0), get_hists()


############ at the moment it is not functional ################
@app.callback(
    Output("train_info", "children"),
    Input("upload-button", "n_clicks")
)
def uiget_path(upload_button):

    # check if button is pushed
    print(upload_button)
    if upload_button:
        subprocess.run("python import_path.py", shell=True)

        # start loading file based on input file_path
        with open("filename.txt", 'r') as file:
            filename = file.read()

        # create path to save
        save_path = os.path.join(os.path.split(filename)[0], "temp_save_add")
        os.makedirs(save_path, exist_ok=True)

        # start processing data
        process_input_data(path_to_file=filename,
                           path_to_save=save_path,
                           start_index=11400,
                           end_index=2900000,
                           epoch_len=10,
                           fr=1000,
                           return_result=False)

    return filename


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
