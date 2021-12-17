# Frontend file

# required libraries
from ntpath import join
from types import LambdaType
from dash_bootstrap_components._components.InputGroup import InputGroup
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb
import os
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# internal files and functions
from utils import process_input_data, read_data_header

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

# university logo path
University_Logo = "https://upload.wikimedia.org/wikipedia/de/9/97/Eberhard_Karls_Universit%C3%A4t_T%C3%BCbingen.svg"

##### Data and parameters storage #####
# data header uploader & downloader


def data_uploader(data, path):

    if isinstance(data, pd.DataFrame):
        data.to_json(path)
    else:
        data = pd.DataFrame(data)
        data.to_json(path)
        print("data first converted to pandas dataframe before save")


def data_downloader(path):

    return pd.read_json(path)


# create dcc store list
all_storage = html.Div([dcc.Store(id="epoch-index"),  # get input from keyboard callback
                        # both below ids are getting updated in toggle_offcanvas
                        dcc.Store(id="input-file-loc"),
                        dcc.Store(id="save-path"),
                        # get input from on_button_click callback
                        dcc.Store(id="user-epoch-length"),
                        dcc.Store(id="user-sampling-frequency"),
                        dcc.Store(id="user-selected-channels"),
                        # getting updated in toggle_offcanvas
                        dcc.Store(id="input-file-default-info"),
                        dcc.Store(id="monitor-info"),
                        # get input from keyboard callback
                        dcc.Store(id="user-pressed-key"),
                        dcc.Store(id="max-possible-epochs"),
                        dcc.Store(id="scoring-labels"),
                        dcc.Store(id="res11"),
                        dcc.Store(id="res12")])
#####       #####       #####       #####       #####

# config menu items
config_menu_items = html.Div(
    [
        dbc.DropdownMenuItem("Human", id="dropdown-menu-item-1"),
        dbc.DropdownMenuItem("Rodent", id="dropdown-menu-item-2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Custom", id="dropdown-menu-item-3"),
    ],
)

# define channels
def define_channels(channel_name=["No Channel in Data"]):

    options = []
    if isinstance(channel_name[0], list):
        channel_name = channel_name[0]

    for i in channel_name:
        options.append({'label': i, 'value': i})

    channels = dbc.Checklist(
        id="channel_checklist",
        options=options,
        value=[],
        switch=True,
        inputStyle={"margin-right": "10px"},
        labelStyle={'display': 'block'},
        style={'color': '#463d3b'}
    )
    return channels


# navigation toolbar with logo, software title and a button
navbar = dbc.NavbarSimple(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(html.A(html.Img(src=University_Logo, height="40px"),
                        href="http://www.physiologie2.uni-tuebingen.de/", target="_blank"), width="auto"),
                dbc.Col(html.H1("Sleezy", style={
                        'color': '#003D7F', 'fontSize': 35})),


                dbc.Col(html.Div(
                    [
                        dbc.Button("Import",
                                   id="import-offcanvas-button", size="sm", n_clicks=0),
                        dbc.Offcanvas(children=[

                            html.P(
                                "1) Please customize the epoch length below:",
                                style={'color': '#463d3b'}
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Epoch length"), dbc.Input(
                                placeholder="in seconds", autocomplete="off", id="epoch-length-input")],
                                class_name="mb-4"),

                            html.P(
                                "2) Please specify the sampling frequency:",
                                style={'color': '#463d3b'}
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Sampling frequency"), dbc.Input(
                                placeholder="in Hz", autocomplete="off", id="sampling_fr_input")],
                                class_name="mb-4"),

                            html.P(
                                "3) Below are the channels of your dataset. "
                                "Please select which ones you want to load, "
                                "and then, press the Load button. "
                                "This can take a couple of minutes!",
                                style={'color': '#463d3b'}
                            ),

                            html.Div(define_channels(), id="channel_def_div"),
                            dbc.Row(dbc.Button("Load", id="load_button", size="sm"),
                                    class_name="mt-3"),

                        ],
                            id="import-offcanvas",
                            title="Before you load, there are 3 steps...",
                            is_open=False,
                            backdrop='static',
                            scrollable=True,
                            style={
                                'title-color': '#463d3b', 'background': 'rgba(224, 236, 240, 0.2)', 'backdrop-filter': 'blur(10px)'}
                        ),
                    ]
                ), width="auto"),

                dbc.Col(dbc.Button("Save", id="save-button",
                        size="sm"), width="auto"),
                dbc.Col(html.Div(
                    [
                        dbc.Button("Advanced",
                                   id="advparam-button", size="sm", n_clicks=0),
                        dbc.Offcanvas(children=[
                            html.Div([
                                dbc.InputGroup([dbc.InputGroupText("ML algorithm"), dbc.Input(
                                    placeholder="Which ML you want to use?")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("nThread"), dbc.Input(
                                    placeholder="The number of dedicated threads")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("GPU"), dbc.Input(
                                    placeholder="1 for yes, 0 for no")], class_name="mb-1"),
                                dbc.InputGroup([dbc.InputGroupText("Lag time"), dbc.Input(
                                    placeholder="How much you can wait for training")], class_name="mb-1")
                            ]),
                            dbc.Row(dbc.Button("Apply", id="apply-params", size="sm"),
                                    class_name="mt-3"),

                        ],
                            id="advparam-offcanvas",
                            title="Here, you can customize the advance parameters!",
                            is_open=False,
                            backdrop='static',
                            scrollable=True,
                            placement='bottom',
                            style={'title-color': '#463d3b', 'background': 'rgba(224, 236, 240, 0.2)', 'backdrop-filter': 'blur(10px)'}
                        ),
                    ]
                ), width="auto"),

                dbc.Col(dbc.Button("About Us", id="about-us-button", size="sm"),
                        width="auto"),
                dbc.Col(dbc.Button("Help", id="help-button",
                        size="sm"), width="auto"),
                dbc.Col(dbc.NavbarToggler(
                    id="navbar-toggler", n_clicks=0), width="auto"),
                dbc.Col(dbc.Collapse(dbc.Input(type="search", placeholder="Search", size="sm"),
                        id="navbar-collapse", is_open=True), width="auto"),
                dbc.Col(dbc.Button("Search epoch", size="sm",
                        n_clicks=0, color="dark"), width="auto")
            ],
            align="center",
            justify="center",
        ),
        fluid=False,
    ),

    links_left=True,
    sticky="top",
    color="info",
    dark=False,
    fluid=False,
    className="mb-3",
)

inputbar = dbc.Nav(children=[
    dbc.Container(children=[
        dbc.Row(
            [
                dbc.Col(
                            [
                                dbc.Input(
                                    max=3,
                                    min=1,
                                    inputmode="numeric",
                                    type="number",
                                    id="minus-one_epoch",
                                    placeholder="",
                                    disabled=True,
                                    style={'width': '100px',
                                           'text-align': 'center'},
                                ),
                            ],
                    class_name="d-flex justify-content-center",
                ),
                dbc.Col(
                    [
                        dbc.Input(
                            max=3,
                            min=1,
                            inputmode="numeric",
                            type="number",
                            id="null_epoch",
                            placeholder="",
                            disabled=True,
                            style={'width': '100px',
                                   'text-align': 'center'},
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),
                
                dbc.Col(
                    [
                        dbc.Input(
                            max=3,
                            min=1,
                            inputmode="numeric",
                            type="number",
                            id="plus-one_epoch",
                            placeholder="",
                            disabled=True,
                            style={'width': '100px',
                                   'text-align': 'center'},
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),
            ]),
            dbc.Row(
            dbc.Col(
                    [
                        dbc.Input(
                            max=3,
                            min=1,
                            inputmode="numeric",
                            # type="number",
                            id="null_epoch_act",
                            placeholder="",
                            autocomplete="off",
                            style={'border': '2px solid', 'border-color': '#003D7F',
                                   'width': '100px', 'text-align': 'center', 'hoverinfo': 'none'},
                            
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),
                )
    ],
        fluid=True,
    )],
    fill=True,
)

sliderbar = dbc.Container(children=[
                dbc.Row(
                    dcc.Slider(
                        id='epoch-sliderbar',
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        tooltip={"placement": "top", "always_visible": True}
                        )
                    )
            ],
            fluid=True,
)

def plot_traces(traces, s_fr=1):

    # traces --> n * p array
    # get trace length
    trace_len = len(traces[:, 0])

    # xaxis
    x_axis = np.linspace(0, trace_len/s_fr, trace_len)

    # number of channels
    nr_ch = traces.shape[1]

    # vertical ines positions
    x0 = (trace_len/3) / s_fr
    x1 = ((2 * trace_len) / 3) / s_fr

    y0 = []
    y1 = []
    y0 = [traces[:, i].min() for i in range(nr_ch)]
    y1 = [traces[:, i].max() for i in range(nr_ch)]

    fig = make_subplots(rows=nr_ch, cols=1, shared_xaxes=True,
                        print_grid=False, vertical_spacing=0.05)

    # changing px.line(y=trace)["data"][0] to go.Scatter(y=trace, mode="lines")
    # increase speed by factor of ~5
    for i in range(nr_ch):
        fig.add_trace(go.Scatter(x=x_axis, y=traces[:, i], mode="lines", line=dict(
            color='#003D7F', width=1)), row=i+1, col=1)

    for i in range(nr_ch):

        # adding lines (alternative is box)
        split_line1 = go.Scatter(x=[x0, x0], y=[y0[i], y1[i]], mode="lines",
                                 hoverinfo='skip',
                                 line=dict(color='black', width=3, dash='6px,3px,6px,3px'))
        split_line2 = go.Scatter(x=[x1, x1], y=[y0[i], y1[i]], mode="lines",
                                 hoverinfo='skip',
                                 line=dict(color='black', width=3, dash='6px,3px,6px,3px'))

        fig.add_trace(split_line1, row=i+1, col=1)
        fig.add_trace(split_line2, row=i+1, col=1)

    fig.update_layout(margin=dict(l=0, r=0, t=1, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False
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
def get_hists(data):

    # list of 2 (power spectrums(by number of channels) and histograms(same))
    # first powerspectrum and then histogram

    spectrums = data[0]  # n*p
    histos = data[1]  # n*p
    nr_ch = histos.shape[1]

    # check if it is True
    assert spectrums.shape[1] == histos.shape[1]

    fig = make_subplots(rows=2, cols=nr_ch, shared_yaxes=True,
                        print_grid=False, vertical_spacing=0.25, horizontal_spacing=0.05,
                        subplot_titles=("Power Spectrums", "", "", "Amplitude Histograms", "", ""))

    for i in range(nr_ch):
        # at the moment x is fixed to 30 Hz 120 = 30 * 4 (always 4)
        fig.add_trace(go.Scatter(x=np.linspace(0, 30, 120), y=spectrums[:, i], mode="lines", line=dict(
            color='black', width=1)), row=1, col=i+1)
        fig.add_trace(go.Histogram(x=histos[:, i], marker_color='LightSkyBlue',
                                   opacity=0.75, histnorm="probability"), row=2, col=i+1)

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


# check user input
def check_user_input(user_input, type):

    try:
        # Convert it into integer
        val = int(user_input)
        if type.lower() == "int":
            return True

    except ValueError:
        try:
            # Convert it into float
            val = float(user_input)
            if type.lower() == "float":
                return True

        except ValueError:
            if type.lower() == "string":
                return True


# start tha main app
# Dash apps are composed of two parts. The first part is the "layout" of the app and it describes what the application looks like.
# The second part describes the interactivity of the application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# storage
Storage = html.Div(dcc.Store(id='storage_add', storage_type='local'))


# Graph div
trace_graphs = html.Div(children=[
    dcc.Graph(id="ch", responsive=True)
])

# lower Row (contains all learning graphs and informations + spectrums and histograms)
# confusion matrix table
# creating df which mimics sklearn output
table = dbc.Container(dbc.Col(dbc.Table.from_dataframe(
    get_confusion_mat(), striped=False, bordered=False, hover=True, index=True, responsive=True, size="md"),
    style={"width": "20vw", "height": "25vh"})
)

# accuracy graph
acc_graph = dcc.Graph(id="accuracy", figure=get_acc_plot(
), responsive=True, style={"width": "20vw", "height": "25vh"})


# hist graphs
hist_graph = dcc.Graph(id="hist-graphs",
                       responsive=True, style={"width": "50vw", "height": "25vh"})

# lower row left-side
lower_row_left = dbc.Container(dbc.Row([
    dbc.Col(table),
    dbc.Col(acc_graph),
    dbc.Col(dbc.Card("Train Info", id="train-info"))
], style={"display": "flex"}, class_name="g-4"), fluid=True)

# lower row right-side
lower_row_right = dbc.Container(dbc.Row([
    hist_graph
]), fluid=True)

# lower row
lower_row = html.Div(dbc.NavbarSimple(html.Div(children=[
    sliderbar,
    dbc.Container(children=[html.H4("Analytics", id="lower-bar",
                                    style={"font-weight": "600", "padding-top": "10px"}),
                            html.Br()],
                  fluid=True,
                  style={"border": "3px solid #308fe3", "backgroundColor": "#8ac6fb"}),
    dbc.Container(dbc.Row([
        dbc.Col(lower_row_left, width=7),
        dbc.Col(lower_row_right, width=5)
    ], class_name="g-4"), fluid=True, style={"backgroundColor": "#Cee7ff"})
]),
    links_left=True,
    fluid=True,
    fixed="bottom",
    style={"border": "0px"},
))


# detecting keyboard keys
my_keyboard = html.Div(Keyboard(id="keyboard"))


# define app layout using dbc container
app.layout = dbc.Container(
    html.Div([all_storage, my_keyboard, navbar, inputbar, trace_graphs, lower_row]), fluid=True)


# all callBacks

# keyboard callback
# 1. reading keyboard keys / epoch-index / max number of possible epoch
# 2. update epoch-index / user pressed key. Both in Storage
@app.callback(
    [Output("epoch-index", "data"),
     Output("user-pressed-key", "data"),
     Output("ch", "figure"),
     Output("hist-graphs", "figure"),
     Output("minus-one_epoch", "value"),
     Output("null_epoch", "value"),
     Output("null_epoch_act", "value"),
     Output("plus-one_epoch", "value"),
     Output("scoring-labels", "data"),
     Output("epoch-sliderbar", "value"),
     Output("epoch-sliderbar", "max")],

    [Input("keyboard", "keydown"),
     Input("keyboard", "n_keydowns"),
     Input("epoch-index", 'data'),
     Input("max-possible-epochs", "data"),
     Input("save-path", "data"),
     Input("user-sampling-frequency", "data"),
     Input("input-file-default-info", "data"),
     Input("import-offcanvas", "is_open"),
     Input("null_epoch_act", "value"),
     Input("scoring-labels", "data"),
     [Input('epoch-sliderbar', 'value')]]
)
def keydown(event, n_keydowns, epoch_index, max_nr_epochs, save_path, user_sample_fr, input_default, off_canvas, score_value, score_storage, slider_value):

    file_exist = False
    if not save_path is None:
        file_exist = os.path.exists(os.path.join(
            save_path, str(0) + ".json"))

    # initialize sampling frequency
    sampling_fr = 1
    if (not input_default is None) and (pd.read_json(input_default).s_freq.values):
        sampling_fr = pd.read_json(input_default).s_freq.values
    if user_sample_fr is not None:
        sampling_fr = int(user_sample_fr)

    # change score_value data type
    if (not score_value is None) and score_value.isnumeric():
        score_value = int(score_value)

    # change score_storage data format
    if not score_storage is None:
        score_storage = pd.read_json(score_storage)
        print("score at the beginning", type(score_storage), score_storage)
    

    # upadte frigures with sliderbar
    if (slider_value != epoch_index):
        score_storage = pd.DataFrame([
            {epoch_index: slider_value}])
            


    # It is important False off_canvas
    if n_keydowns and file_exist and not off_canvas:

        # change input types
        epoch_index = int(epoch_index)
        if max_nr_epochs is None:
            max_nr_epochs = 10000  # temp
        else:
            max_nr_epochs = int(max_nr_epochs)

        max_sliderbar = max_nr_epochs
        # marks_slidebar = {i: str(i) for i in range(1, max_sliderbar, int(max_sliderbar/10))}
        print("Epoch index at the beginning", epoch_index)

        # update figures with only left/right arrow keys
        if ((event["key"] == "ArrowRight") or (event["key"] == "ArrowLeft")):

            # check what is user pressed key
            if (event["key"] == "ArrowRight"):
                if epoch_index < max_nr_epochs:
                    epoch_index += 1

            elif (event["key"] == "ArrowLeft"):
                if epoch_index > 0:
                    epoch_index -= 1
            
            slider_value = epoch_index
                    

        # update figures with score labels
        if score_value == 1 or score_value == 2 or score_value == 3:

            # saving score label to storage
            if not score_storage is None:
                score_storage = pd.concat([score_storage, pd.DataFrame(
                    [{epoch_index: score_value}])], axis=1)
            else:
                score_storage = pd.DataFrame([
                    {epoch_index: score_value}])

            if epoch_index < max_nr_epochs:
                epoch_index += 1

            slider_value = epoch_index


        # read data batch from disk / check if save_path exist
        df_mid = pd.read_json(os.path.join(
            save_path, str(epoch_index) + ".json"))
        data_mid = np.stack(df_mid["data"])
        ps_mid = np.stack(df_mid["spectrums"]).T
        hist_mid = np.stack(df_mid["histograms"]).T

        full_ps_hist = [ps_mid, hist_mid]

        if epoch_index == max_nr_epochs:
            data_right = np.zeros_like(data_mid)
        else:
            data_right = np.stack(pd.read_json(os.path.join(
                save_path, str(epoch_index + 1) + ".json"))["data"])

        if epoch_index == 0:
            data_left = np.zeros_like(data_mid)
        else:
            data_left = np.stack(pd.read_json(os.path.join(
                save_path, str(epoch_index - 1) + ".json"))["data"])

        # combine mid_right datasets
        full_trace = np.hstack(
            [data_left,
                data_mid,
                data_right])

        # call for plot functions
        fig_traces = plot_traces(full_trace.T, s_fr=sampling_fr)
        ps_hist_fig = get_hists(data=full_ps_hist)
        print("Epoch index after plot", epoch_index)

        # check and update score labels (after key left/right if they exist)
        if not score_storage is None:
            # pdb.set_trace()
            if epoch_index in score_storage.keys():
                null_score_label = str(
                    score_storage[epoch_index].values[0])
            else:
                null_score_label = ""

            if (epoch_index - 1) in score_storage.keys():
                epoch_minus_one_label = str(
                    score_storage[epoch_index - 1].values[0])
            else:
                epoch_minus_one_label = ""

            if (epoch_index + 1) in score_storage.keys():
                epoch_plus_one_label = str(
                    score_storage[epoch_index + 1].values[0])
            else:
                epoch_plus_one_label = ""

        else:
            null_score_label = ""
            epoch_minus_one_label = ""
            epoch_plus_one_label = ""

        # change datatype
        if not score_storage is None:
            # pdb.set_trace()
            score_storage = score_storage.to_json()
        
        

        return epoch_index, event, fig_traces, ps_hist_fig, epoch_minus_one_label, null_score_label, "", epoch_plus_one_label, score_storage, slider_value, max_sliderbar

    return json.dumps(0), None, plot_traces(np.zeros((1000, 1))), get_hists([np.zeros((1000, 1)), np.zeros((1000, 1))]), "", "", "", "", None, None, None


# collapse callback
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],


)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# This part has to merge to above callback after fixing issue
# reading user input for epoch len (custom value)
@app.callback(
    Output("user-epoch-length", "data"),
    Input("epoch-length-input", "value")
)
def user_custom_epoch_length(value):
    return value


# Parameters collapse
@ app.callback(
    Output("advparam-offcanvas", "is_open"),
    [Input("advparam-button", "n_clicks")],
    [State("advparam-offcanvas", "is_open")],
)
def toggle_adv_param_offcanvas(n, is_open):
    if n:
        return not is_open
    return is_open


# channels loading callback
@ app.callback(
    [Output("import-offcanvas", "is_open"),
     Output("input-file-loc", "data"),
     Output("save-path", "data"),
     Output("channel_def_div", "children"),
     Output("input-file-default-info", "data")],

    Input("import-offcanvas-button", "n_clicks"),

    [State("import-offcanvas", "is_open")],
)
def toggle_import_offcanvas(n1, is_open):
    if n1:
        # reading only data header and updating Import button configs
        subprocess.run("python import_path.py", shell=True)

        # read input path from text file already generated
        with open("filename.txt", 'r') as file:
            filename = file.read()

        # create path to temporal saves
        save_path = os.path.join(os.path.split(filename)[0], "temp_save_add")
        os.makedirs(save_path, exist_ok=True)

        # start reading data header (output of the file is a dataframe)
        data_header = read_data_header(filename)

        # I need to run the define_channels function
        channel_children = define_channels(
            channel_name=data_header["channel_names"])

        # save header file to disk
        data_uploader(data=data_header,
                      path=os.path.join(save_path, "header_data.json"))

        # button canvas, input-data-path, save-path, channel name, save-data-header
        return not is_open, filename, save_path, channel_children, data_header.to_json()
    return is_open, None, None, "No Input Channels", None




@ app.callback(
    Output("user-sampling-frequency", "data"),
    Input("sampling_fr_input", "value"),
)
def handle_sample_fr_input(value):
    return value


@app.callback(
    Output("user-selected-channels", "data"),
    Input("channel_checklist", "value")
)
def get_channel_user_selection(channels):
    return json.dumps(channels)


@app.callback(
    [Output("load_button", "children"),
     Output("max-possible-epochs", "data")],
    [Input("load_button", "n_clicks"),
     Input("input-file-loc", "data"),
     Input("save-path", "data"),
     Input("user-epoch-length", "data"),
     Input("user-sampling-frequency", "data"),
     Input("user-selected-channels", "data")]
)
def action_load_button(n, filename, save_path, epoch_len, sample_fr, channel_list):

    if n:
        print(
            f"\n\nfilename: {filename} \nsavepath: {save_path} \nepoch_len:{epoch_len} \nsampling_fr:{sample_fr} \n\n")
        max_epoch_nr = process_input_data(path_to_file=filename,
                                          path_to_save=save_path,
                                          start_index=0,
                                          end_index=-1,
                                          epoch_len=int(epoch_len),
                                          fr=int(sample_fr),
                                          channel_list=json.loads(
                                              channel_list),
                                          return_result=False)
        print("Finished data loading")
        print(f"Max epochs: {max_epoch_nr}")
        # [dbc.Spinner(size="sm"), " Loading..."]
        return "Loaded", json.dumps(max_epoch_nr)
    else:
        return "Load", None

# open browser
#chrome_options = Options()
#chrome_options.add_argument("--kiosk")
#driver = webdriver.Chrome(chrome_options=chrome_options)
#driver.get('http://localhost:8050/')

@app.callback(
    Output("save-button", "children"),

    [Input("save-button", "n_clicks"),
     Input("input-file-loc", "data"),
     Input("scoring-labels", "data")]
)
def save_button(n_clicks, input_data_loc, scoring_results):

    if n_clicks:
        print("inside save button")
        # first create a folder or make sure the folder exist
        save_path = os.path.join(os.path.split(
            input_data_loc)[0], "SleezyResults")
        os.makedirs(save_path, exist_ok=True)

        # saving scoring results
        #   1. reading as pandas dataframe
        scoring_results = pd.read_json(scoring_results)

        #   2. saving in any suitable format
        scoring_results.to_json(os.path.join(save_path, "score_results.json"))

        scoring_results.to_csv(os.path.join(
            save_path, "score_results.csv"), index=False)

        return "Save"
    return "Save"


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
