# Frontend file

# required libraries
from dash_bootstrap_components._components.InputGroup import InputGroup
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb
import os
import subprocess

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
                        dcc.Store(id="res10"),
                        dcc.Store(id="res11"),
                        dcc.Store(id="res12")])
#####       #####       #####       #####       #####


# draft training parameters list
input_params = html.Div([
    dbc.InputGroup([dbc.InputGroupText("ML algorithm"), dbc.Input(
        placeholder="Which ML you want to use?")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("nThread"), dbc.Input(
        placeholder="The number of dedicated threads")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("GPU"), dbc.Input(
        placeholder="1 for yes, 0 for no")], class_name="mb-1"),
    dbc.InputGroup([dbc.InputGroupText("Lag time"), dbc.Input(
        placeholder="How much you can wait for training")], class_name="mb-1")
])

# config menu items
config_menu_items = html.Div(
    [
        dbc.DropdownMenuItem("Human", id="dropdown-menu-item-1"),
        dbc.DropdownMenuItem("Rodent", id="dropdown-menu-item-2"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Custom", id="dropdown-menu-item-3"),
    ],
)

# import configs
input_config = html.Div([
    dbc.InputGroup([
        dbc.DropdownMenu(config_menu_items, label="Species"),
        dbc.Input(id="epoch-length-input", placeholder="Epoch length")]),
    dbc.InputGroup([dbc.InputGroupText("Channels to import"), dbc.Input(
        placeholder="How many channels do you have?")])
])

# advanced parameters button
param_collapse = html.Div([
    dbc.Button("Advanced parameters", id="param_collapse_button",
               size="sm", n_clicks=0),
    dbc.Collapse(input_params, id="param_collapse", is_open=False)
])


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
    )
    return channels


# navigation toolbar with logo, software title and a button
navbar = dbc.NavbarSimple(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(html.A(html.Img(src=University_Logo, height="40px"),
                        href="http://www.physiologie2.uni-tuebingen.de/"), width="auto"),
                dbc.Col(html.H1("Sleezy", style={
                        'color': '#003D7F', 'fontSize': 35})),


                dbc.Col(html.Div(
                    [
                        dbc.Button("Import",
                                   id="open-offcanvas", size="sm", n_clicks=0),
                        dbc.Offcanvas(children=[

                            html.P(
                                "1) Please customize the epoch length below:"
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Epoch length"), dbc.Input(
                                placeholder="in seconds", autocomplete="off", id="epoch-length-input")],
                                class_name="mb-4"),

                            html.P(
                                "2) Please specify the sampling frequency:"
                            ),
                            dbc.InputGroup([dbc.InputGroupText("Sampling frequency"), dbc.Input(
                                placeholder="in Hz", autocomplete="off", id="sampling_fr_input")],
                                class_name="mb-4"),

                            html.P(
                                "3) Below are the channels of your dataset. "
                                "Please select which ones you want to load, "
                                "and then, press the Load button. "
                                "This can take a couple of minutes!"
                            ),

                            html.Div(define_channels(), id="channel_def_div"),
                            dbc.Row(dbc.Button("Load", id="load_button", size="sm"),
                                    class_name="mt-2"),

                        ],
                            id="offcanvas",
                            title="Before you load, there are 3 steps...",
                            is_open=False,
                            backdrop='static',
                            scrollable=True
                        ),
                    ]
                ), width="auto"),

                dbc.Col(dbc.Button("Save", id="save-button",
                        size="sm"), width="auto"),
                dbc.Col(dbc.Button(
                    "Advanced", id="param_collapse_button", size="sm", n_clicks=0), width="auto"),
                dbc.Col(dbc.Collapse(input_params, id="param_collapse",
                        is_open=False), width="auto"),
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
    dbc.Container(
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
                            style={
                                'width': '100px', 'text-align': 'center', 'hoverinfo': 'none'},
                        ),
                        dbc.Col(
                            [
                                dbc.Input(
                                    max=3,
                                    min=1,
                                    inputmode="numeric",
                                    # type="number",
                                    id="null_epoch",
                                    placeholder="",
                                    autocomplete="off",
                                    style={'border': '2px solid', 'border-color': '#003D7F',
                                           'width': '100px', 'text-align': 'center', 'hoverinfo': 'none'},
                                ),
                            ],
                            class_name="d-flex justify-content-center",
                        ),
                    ],
                    class_name="d-flex justify-content-center",
                ),
            ]),
        fluid=True,
    )],
    fill=True,
)


def plot_traces(traces):

    # traces --> n * p array
    # get trace length
    trace_len = len(traces[:, 0])

    # number of channels
    nr_ch = traces.shape[1]

    # vertical ines positions
    x0 = trace_len/3
    x1 = (2 * trace_len) / 3

    y0 = []
    y1 = []
    y0 = [traces[:, i].min() for i in range(nr_ch)]
    y1 = [traces[:, i].max() for i in range(nr_ch)]

    fig = make_subplots(rows=nr_ch, cols=1, shared_xaxes=True,
                        print_grid=False, vertical_spacing=0.05)

    # changing px.line(y=trace)["data"][0] to go.Scatter(y=trace, mode="lines")
    # increase speed by factor of ~5
    for i in range(nr_ch):
        fig.add_trace(go.Scatter(y=traces[:, i], mode="lines", line=dict(
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
        fig.add_trace(go.Scatter(y=spectrums[:, i], mode="lines", line=dict(
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
     Output("hist-graphs", "figure")],

    [Input("keyboard", "keydown"),
     Input("keyboard", "n_keydowns"),
     Input("epoch-index", 'data'),
     Input("max-possible-epochs", "data"),
     Input("save-path", "data")]
)
def keydown(event, n_keydowns, epoch_index, max_nr_epochs, save_path):

    file_exist = False
    if not save_path is None:
        file_exist = os.path.exists(os.path.join(
            save_path, str(0) + ".json"))

    if n_keydowns and file_exist:

        # change input types
        epoch_index = int(epoch_index)
        max_nr_epochs = 10000  # temp
        print("Epoch index at the beginning", epoch_index)
        # check what is user pressed key
        if (event["key"] == "ArrowRight"):

            if epoch_index < max_nr_epochs:
                epoch_index += 1

        elif (event["key"] == "ArrowLeft"):
            if epoch_index > 0:
                epoch_index -= 1

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
        fig_traces = plot_traces(full_trace.T)
        ps_hist_fig = get_hists(data=full_ps_hist)
        print("Epoch index after plot", epoch_index)

        return epoch_index, event, fig_traces, ps_hist_fig

        # return epoch_index, event, fig_traces
    elif (not n_keydowns) and (not save_path is None) and file_exist:
        # read data batch from disk / check if save_path exist
        df_mid = pd.read_json(os.path.join(
            save_path, str(0) + ".json"))
        data_mid = np.stack(df_mid["data"])

        ps_mid = np.stack(df_mid["spectrums"]).T
        hist_mid = np.stack(df_mid["histograms"]).T

        df_right = pd.read_json(os.path.join(
            save_path, str(1) + ".json"))
        data_right = np.stack(df_right["data"])

        full_ps_hist = [ps_mid, hist_mid]
        # combine mid_right datasets
        full_trace = np.hstack(
            [np.zeros_like(data_mid),
             data_mid,
             data_right])

        fig_traces = plot_traces(full_trace.T)
        ps_hist_fig = get_hists(data=full_ps_hist)

        return json.dumps(0), None, fig_traces, ps_hist_fig

    return json.dumps(0), None, plot_traces(np.zeros((1000, 1))), get_hists([np.zeros((1000, 1)), np.zeros((1000, 1))])


# collapse callback
@ app.callback(
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
    Output("param_collapse", "is_open"),
    [Input("param_collapse_button", "n_clicks")],
    [State("param_collapse", "is_open")],
)
def toggle_param_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# epoch scorings callback
# NOT FUNCTIONAL FOR NOW
@ app.callback(
    [Output("minus-one_epoch", "value"),
     Output("null_epoch", "value")],
    [Input("null_epoch", "value")])
def output_text(value):
    print(type(value))
    if value == 1 or value == 2 or value == 3:
        value = value
        return value, ""
    else:
        value = value
        return value, ""


# channels loading callback
@ app.callback(
    [Output("offcanvas", "is_open"),
     Output("input-file-loc", "data"),
     Output("save-path", "data"),
     Output("channel_def_div", "children"),
     Output("input-file-default-info", "data")],

    Input("open-offcanvas", "n_clicks"),

    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
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
    Output("load_button", "children"),
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
        process_input_data(path_to_file=filename,
                           path_to_save=save_path,
                           start_index=0,
                           end_index=-1,
                           epoch_len=int(epoch_len),
                           fr=int(sample_fr),
                           channel_list=json.loads(channel_list),
                           return_result=False)
        print("Finished data loading")
        return [dbc.Spinner(size="sm"), " Loading..."]
    else:
        return "Load"


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
