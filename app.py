# Frontend file

# required libraries
from datetime import time
from ntpath import join
from types import LambdaType
#from wsgiref.types import InputStream
from dash_bootstrap_components._components.InputGroup import InputGroup
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import json
import pdb
import time
import os
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options

# internal files and functions
from utils import process_input_data, read_data_header
from models import Classifier

# dash library
import dash
from dash import dcc
from dash import html
#import dash_daq as daq
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, MATCH, ALL
import plotly.express as px
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate

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
                        dcc.Store(id="user-selected-indexes"),
                        # getting updated in toggle_offcanvas
                        dcc.Store(id="input-file-default-info"),
                        dcc.Store(id="monitor-info"),
                        # get input from keyboard callback
                        dcc.Store(id="user-pressed-key"),
                        dcc.Store(id="max-possible-epochs"),
                        dcc.Store(id="scoring-labels"),
                        dcc.Store(id="slider-saved-value"),
                        dcc.Store(id="AI-accuracy"),
                        dcc.Store(id="AI-trigger-params")])
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


# define channels & dropdowns
def define_channels(channel_name=["No Channel in Data"], disabled=False, value=[]):
    options = []
    dropdowns = []
    mins = []
    maxes = []
    if isinstance(channel_name[0], list):
        channel_name = channel_name[0]

    for nr, i in enumerate(channel_name):
        options.append({'label': i, 'value': i, 'disabled': disabled})
        dropdowns.append(dbc.Select(
                placeholder='N/A',
                options=[
                    {"label": "Raw Data", "value": "raw"},
                    {"label": "Filtered", "value": "flt"},
                    {"label": "Both!", "value": "bth"},
                ],
                id={"type":"ddowns", "index":nr},
                disabled=True,
                style={"width": "110px", 'filter':'blur(150px)', 'opacity':'0'},
                class_name = 'mb-2',
                size="sm",
                ))
        mins.append(dbc.Input(placeholder="Min", size="sm", id={"type":"mins", "index":nr}, disabled=True, 
                style={"width": "80px", 'filter':'blur(150px)', 'opacity':'0'}, class_name = 'mb-2', inputmode = 'numeric', type="number",
                min = 0, max = 1000))
        maxes.append(dbc.Input(placeholder="Max", size="sm", disabled=True,
                style={"width": "80px", 'filter':'blur(150px)', 'opacity':'0'}, class_name = 'mb-2', id={"type":"maxes", "index":nr},
                inputmode = 'numeric', type="number", min = 0, max = 1000))

    components = dbc.Row(children=[
            dbc.Col(dbc.Checklist(
            id="channel_checklist",
            options=options,
            value=value,
            switch=True,
            inputStyle={"margin-right": "0px"},
            labelStyle={'display': 'block'},
            label_class_name = 'mb-3',
            style={'width':'110px', 'color': '#463d3b'}
                    ),
                ),
            
            dbc.Col(dropdowns, class_name = 'me-4'),
            dbc.Col(mins, class_name = 'px-0 g-0 mx-0'),
            dbc.Col(maxes, class_name = 'px-0 g-0 mx-0'),
            ])
    
    return components


# navigation toolbar with logo, software title and a button
navbar = dbc.NavbarSimple(
    dbc.Container(
        dbc.Row(
            [

                dbc.Col(html.H1("Sleezy!", style={'margin-left': '100px',
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

                            html.Div(children=define_channels(),
                                     id="channel_def_div"),
                                  
                            dbc.Row(children=[dbc.Button("Load", id="load_button", size="sm", n_clicks=0),
                                    html.Div(children=False, id="second_execution")],
                                    class_name="mt-3"),
                            html.Br(),
                            dcc.Loading(id="loading-state",
                                        children=html.Div(id="loading-output")),
                            dcc.Interval(
                                id='internal-trigger', interval=100, n_intervals=0, max_intervals=0),

                        ],
                            id="import-offcanvas",
                            title="Before you load, there are 3 steps...",
                            is_open=False,
                            backdrop='static',
                            scrollable=True,
                            style={'width' : '500px',
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
                            style={
                                'title-color': '#463d3b', 'background': 'rgba(224, 236, 240, 0.2)', 'backdrop-filter': 'blur(10px)'}
                        ),
                    ]
                ), width="auto"),
                dbc.Col(html.A(dbc.Button("About Us", id="about-us-button", size="sm"), href="http://www.physiologie2.uni-tuebingen.de/", target="_blank"),
                        width="auto", style={'margin-left': '300px'}),
                dbc.Col(html.A(dbc.Button("Help", id="help-button", size="sm"),
                        href="https://github.com/NimaMojtahedi/Automatic-sleep-EEG-scoring", target="_blank"), width="auto"),
                dbc.Col(html.A(html.Img(src=University_Logo, height="40px"),
                        href="http://www.physiologie2.uni-tuebingen.de/", target="_blank"), width="auto"),
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
                                    style={'width': '80px',
                                           'text-align': 'center'},
                                    size='sm'
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
                            style={'width': '80px',
                                   'text-align': 'center'},
                            size='sm'
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
                            style={'width': '80px',
                                   'text-align': 'center'},
                            size='sm'
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
                               'width': '80px', 'text-align': 'center', 'hoverinfo': 'none'},
                        size='sm'

                    ),
                ],
                class_name="d-flex justify-content-center",
            ),
        ),
    ],
        fluid=True
    )],
    fill=True,
    pills=True,
    navbar=True,
    class_name="sticky-top",
)

graph_bar = dbc.Nav(dbc.Container(dbc.Row(
    dcc.Graph(id="ch", responsive=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['zoomin', 'zoomout', 'zoom', 'pan'],
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'scrollZoom': True,
        'editable': False,
        'showLink': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'EEG-Plot',
            'width': 3200,
            'scale': 1,
        }
    })
),
    class_name="fixed-bottom",
    style={"padding": "25px", "margin-bottom": "300px",
           "width": "100%", "height": "620px"},
    fluid=True))

sliderbar = dbc.Container(children=[
    dbc.Row(
        dcc.Slider(
            id='epoch-sliderbar',
            min=0,
            max=10,
            step=1,
            value=1,
            tooltip={"placement": "top", "always_visible": True}
        )
    )
],
    fluid=True,
)


def plot_traces(traces, names='Null Channel', s_fr=1):
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
            color='#003D7F', width=1), hoverinfo='skip'), row=i+1, col=1)

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
                          showlegend=False,
                          xaxis_fixedrange=True,
                          )
        fig['layout'].update({'yaxis{}'.format(i+1): dict(title=names[i])})
    return fig


# get accuracy plot
def get_acc_plot(data):
    # start plotting
    fig = px.line(y=data)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=0, b=0),
                      font={'size': 10, 'color': '#003D7F'},
                      xaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Iterations", "showgrid": True, "showline": True},
                      yaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Accuracy (%)", "showgrid": True, "showline": True},
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True,
                      showlegend=False,
                      )

    return fig


# spectrum & histograms
def get_hists(data, names='Null'):
    # list of 2 (power spectrums(by number of channels) and histograms(same))
    # first powerspectrum and then histogram

    spectrums = data[0]  # n*p
    histos = data[1]  # n*p
    nr_ch = histos.shape[1]

    # check if it is True
    assert spectrums.shape[1] == histos.shape[1]

    fig = make_subplots(rows=1, cols=nr_ch*2,
                        print_grid=False, vertical_spacing=0.0, horizontal_spacing=0.03,
                        #subplot_titles=("Power Spectrums", "", "", "Amplitude Histograms", "", "")
                        )
    # my_layout=dict(xaxis={"automargin":True,"title_standoff":0, "gridcolor":'rgba(0,61,127,0.2)',"linewidth":2, "linecolor": '#003D7F',"tickfont": {'size':12, 'color': '#003D7F'}, "title": "Iterations","showgrid": True, "showline": True})
    for i in range(nr_ch):
        # at the moment x is fixed to 30 Hz 120 = 30 * 4 (always 4)
        fig.add_trace(go.Scatter(x=np.linspace(0, 30, 120), y=spectrums[:, i], mode="lines", line=dict(
            color='black'), hoverinfo='skip'), row=1, col=i+1)
        fig.add_trace(go.Histogram(x=histos[:, i], marker_color='LightSkyBlue',
                                   opacity=0.75, histnorm="probability", hoverinfo='skip'), row=1, col=nr_ch+i+1)

    # in case it is necessary for histograms xbins=dict(start=-3.0,end=4,size=0.5)
    fig.update_layout(margin=dict(l=1, r=1, t=1, b=1),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', bargap=0.25,
                      font={'size': 10, 'color': '#003D7F'},
                      xaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Frequency (Hz)", "showgrid": True, "showline": True},
                      yaxis={"automargin": True, "title_standoff": 0, "gridcolor": 'rgba(0,61,127,0.2)', "linewidth": 2, "linecolor": '#003D7F', "tickfont": {
                          'size': 12, 'color': '#003D7F'}, "title": "Spectral density", "showgrid": True, "showline": True},
                      showlegend=False,
                      xaxis_fixedrange=True,
                      yaxis_fixedrange=True,
                      )

    fig.update_yaxes(fixedrange=True, gridcolor='rgba(0,61,127,0.2)', linewidth=2, linecolor='#003D7F', tickfont={
                     'size': 12, 'color': '#003D7F'}, showgrid=True, showline=True)
    fig.update_xaxes(fixedrange=True, gridcolor='rgba(0,61,127,0.2)', linewidth=2, linecolor='#003D7F', tickfont={
                     'size': 12, 'color': '#003D7F'}, showgrid=True, showline=True)

    for i in range(nr_ch):
        fig['layout'].update({'xaxis{}'.format(i+1): dict(title=names[i])})
        fig['layout'].update(
            {'xaxis{}'.format(nr_ch+i+1): dict(title=names[i])})

    return fig


def get_confusion_mat(y_true, y_pred, class_names):

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    df = pd.DataFrame(np.round(cm, 2), columns=class_names, index=class_names)

    return dbc.Table.from_dataframe(df, striped=False, bordered=False, hover=True, index=True, responsive=True, size="sm", color="info", style={'color': '#003D7F', 'font-size': 14})


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB], suppress_callback_exceptions=True)

# storage
Storage = html.Div(dcc.Store(id='storage_add', storage_type='local'))


# background for the lower row
backgrd = html.Div(
    dbc.Container(style={"padding": "0", "margin": "0", "width": "100%", "height": "300px", "border": "0px solid #308fe3", "background-image": "url(https://previews.123rf.com/images/gonin/gonin1710/gonin171000004/87977156-blue-white-gradient-hexagons-turned-abstract-background-with-geometrical-elements-modern-3d-renderin.jpg)",
                         'opacity': '0.15', 'filter': 'blur(20px)', "background-size": "100% 100%"},
                  fluid=True,
                  class_name="fixed-bottom",
                  ))

# lower row (contains all learning graphs and informations + spectrums and histograms)
conf_matrix_contents = html.Div(children=get_confusion_mat(np.array(
    [0, 1, 2]), np.array([1, 1, 2]), ['1', '2', '3']), id="table-contents")

lower_row = dbc.Nav(dbc.Container(children=[
    sliderbar,
    dbc.Container(dbc.Row(children=[html.H4("Analytics", id="lower-bar-title",
                                            style={"font-weight": "600", "padding-top": "0px"}),
                                    ]),
                    fluid=True,
                    ),

    html.Div([dbc.Row([
        dbc.Col(html.H6("Confusion Matrix",
                        style={"font-weight": "600", "margin-left": "15px"}), width={"size": 2}),
        dbc.Col(html.H6("AI Acuuracy",
                        style={"font-weight": "600", "margin-left": "0px"}), width={"size": 1, "offset": 0}),
        dbc.Col(html.H6("Power Spectrums",
                        style={"font-weight": "600", "margin-left": "65px"}), width={"size": 2, "offset": 0}),
        dbc.Col(html.H6("Amplitude Histograms",
                        style={"font-weight": "600", "margin-left": "120px"}), width={"size": 3, "offset": 2}),
    ]),
    ]),

    dbc.Nav(dbc.Container(dbc.Row([
        dbc.Container(dbc.Col(conf_matrix_contents),
            style={"width": "220px", "height": "150px", "padding": "0px"}),

        dbc.Container(dbc.Col(dcc.Graph(id="accuracy", figure=get_acc_plot(data=np.array(0)),
                                        responsive=True, style={"width": "200px", "height": "150px"}, config={
            'displayModeBar': False,
            'displaylogo': False,
            'scrollZoom': False,
            'editable': False,
            'showLink': False,
        })),
            style={"width": "200px", "height": "150px"}),

        dbc.Container(dbc.Col(dcc.Graph(id="hist-graphs", responsive=True,
                                        style={"width": "1350px", "height": "130px"}, config={
                                            'displayModeBar': True,
                                            'modeBarButtonsToRemove': ['zoomin', 'zoomout', 'zoom', 'pan', 'select', 'lasso2d', 'autoscale'],
                                            'displaylogo': False,
                                            'scrollZoom': False,
                                            'editable': False,
                                            'showLink':False,
                                            'toImageButtonOptions': {
                                                'format': 'png',
                                                'filename': 'PowerSpect_Histograms',
                                                'width': 1600,
                                                'scale': 1,
                                            }})),
                      style={"width": "1350px", "height": "130px"})
    ],

    ),
        fluid=True,
        style={"margin-top": "0px"}
    ),
        fill=True),

    dbc.Container(dbc.Row(
        dbc.Input(placeholder="  Training information", id="train-info", disabled=True, size='sm',
                  style={"padding": "0", "margin": "0"})),
                  fluid=True,
                  class_name="g-0 mb-0 p-0",
                  style={"margin-top": "0px"}
                  )
],

    fluid=True,
    style={"border": "0px", "width": "100%", "height": "300px"},
),
    fill=True,
    class_name="fixed-bottom",
    # links_left=True,
    #color= 'rgba(224, 236, 240, 0.3)',
    # fixed="bottom",
)

# detecting keyboard keys
my_keyboard = html.Div(Keyboard(id="keyboard"))


# define app layout using dbc container
app.layout = dbc.Container(
    html.Div([all_storage, my_keyboard, navbar, inputbar, graph_bar, backgrd, lower_row]), fluid=True)


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
     Output("epoch-sliderbar", "max"),
     Output("slider-saved-value", "data"),
     Output("AI-trigger-params", "data")],

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
     Input('epoch-sliderbar', 'value'),
     Input("slider-saved-value", "data"),
     Input("user-selected-channels", "data")]
)
def keydown(event, n_keydowns, epoch_index, max_nr_epochs, save_path, user_sample_fr, input_default, off_canvas, score_value, score_storage, slider_live_value, slider_saved_value, channel_list):

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

    # It is important False off_canvas
    if (n_keydowns and file_exist and not off_canvas) or ((slider_live_value != slider_saved_value) and file_exist and not off_canvas and (not slider_live_value is None) and (not slider_saved_value is None)):

        # change input types
        epoch_index = int(epoch_index)
        if max_nr_epochs is None:
            max_nr_epochs = 10000  # temp
        else:
            max_nr_epochs = int(max_nr_epochs)

        max_sliderbar = max_nr_epochs
        # marks_slidebar = {i: str(i) for i in range(1, max_sliderbar, int(max_sliderbar/10))}
        # there is change in slider value
        if (slider_saved_value != slider_live_value) and (not slider_live_value is None) and (not slider_saved_value is None):
            # update epoch to current slider value
            epoch_index = int(slider_live_value)

        # update figures with only left/right arrow keys
        if ((event["key"] == "ArrowRight") or (event["key"] == "ArrowLeft")):

            # check what is user pressed key
            if (event["key"] == "ArrowRight"):
                if epoch_index < max_nr_epochs:
                    epoch_index += 1

            elif (event["key"] == "ArrowLeft"):
                if epoch_index > 0:
                    epoch_index -= 1

            slider_live_value = epoch_index

        # update figures with score labels
        if score_value == 1 or score_value == 2 or score_value == 3:

            # saving score label to storage
            if not score_storage is None:
                # re-scoring effect
                if epoch_index in score_storage.keys():
                    score_storage[epoch_index] = score_value
                else:
                    score_storage = pd.concat([score_storage, pd.DataFrame(
                        [{epoch_index: score_value}])], axis=1)
            else:
                score_storage = pd.DataFrame([
                    {epoch_index: score_value}])

            if epoch_index < max_nr_epochs:
                epoch_index += 1

            slider_live_value = epoch_index

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
        fig_traces = plot_traces(full_trace.T, names=json.loads(
            channel_list), s_fr=sampling_fr)
        ps_hist_fig = get_hists(
            data=full_ps_hist, names=json.loads(channel_list))
        print("The current epoch index is ", epoch_index)

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
            score_storage = score_storage.to_json()

        # check epoch and score_storage to trigger ml train
        if (not epoch_index is None) and (not score_storage is None) and (epoch_index % 10 == 0) and (epoch_index > 0):
            ml_trigger = pd.DataFrame({"epoch_index": [epoch_index],
                                       "score_storage": [score_storage],
                                       "save_path": save_path}).to_json()
        else:
            ml_trigger = dash.no_update

        return epoch_index, event, fig_traces, ps_hist_fig, epoch_minus_one_label, null_score_label, "", epoch_plus_one_label, score_storage, slider_live_value, max_sliderbar, epoch_index, ml_trigger

    return json.dumps(0), None, plot_traces(np.zeros((1000, 1))), get_hists([np.zeros((1000, 1)), np.zeros((1000, 1))]), "", "", "", "", None, None, None, None, dash.no_update


# This part has to merge to above callback after fixing issue
# reading user input for epoch len (custom value)
@app.callback(
    Output("user-epoch-length", "data"),
    [Input("epoch-length-input", "value")]
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


# secondary execution call-back
# channels importing and loading callback
@ app.callback(
    [Output("import-offcanvas", "is_open"),
     Output("input-file-loc", "data"),
     Output("save-path", "data"),
     Output("channel_def_div", "children"),
     Output("input-file-default-info", "data"),
     Output("load_button", "children"),
     Output("max-possible-epochs", "data"),
     Output("loading-output", "children"),
     Output("load_button", "disabled"),
     Output("sampling_fr_input", "disabled"),
     Output("epoch-length-input", "disabled"),
     Output("import-offcanvas-button", "n_clicks"),
     Output("load_button", "n_clicks"),
     Output("second_execution", "children"),
     Output("internal-trigger", "max_intervals"),
     Output("internal-trigger", "n_intervals")
     ],

    [Input("import-offcanvas-button", "n_clicks"),
     Input("load_button", "n_clicks"),
     Input("input-file-loc", "data"),
     Input("save-path", "data"),
     Input("second_execution", "children"),
     Input("internal-trigger", "n_intervals")],

    [State("user-epoch-length", "data"),
     State("user-sampling-frequency", "data"),
     State("user-selected-channels", "data")]
)
def toggle_import_load_offcanvas(n1, n2, filename, save_path, secondary, self_trigger, epoch_len, sample_fr, channel_list):

    if secondary == True and self_trigger == 1:
        secondary = False
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
        print(f"Max epochs: {max_epoch_nr}")
        return False, filename, save_path, dash.no_update, dash.no_update, "Loaded Successfully!", json.dumps(max_epoch_nr), "", True, True, True, 0, 0, secondary, 0, 0

    elif n2:
        n2 = n2 - 1
        data_header = read_data_header(filename)
        channel_children = define_channels(
            channel_name=data_header["channel_names"], disabled=True, value=json.loads(channel_list))
        secondary = True
        return dash.no_update, dash.no_update, dash.no_update, channel_children, dash.no_update, "Loading...", dash.no_update, dash.no_update, True, True, True, 0, 0, secondary, 1, 0

    elif n1 != n2:
        n1 = n1 - 1
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
            channel_name=data_header["channel_names"], disabled=False, value=[])

        # save header file to disk
        data_uploader(data=data_header,
                      path=os.path.join(save_path, "header_data.json"))

        # button canvas, input-data-path, save-path, channel name, save-data-header
        return True, filename, save_path, channel_children, data_header.to_json(), "Load", None, dash.no_update, False, False, False, n1, n2, dash.no_update, dash.no_update, dash.no_update

    else:
        raise PreventUpdate


@ app.callback(
    Output("user-sampling-frequency", "data"),
    [Input("sampling_fr_input", "value")]
)
def handle_sample_fr_input(value):
    return value


@app.callback(
    [Output("user-selected-channels", "data"),
    Output("user-selected-indexes", "data")],
    [Input("channel_checklist", "value"),
    Input("input-file-loc", "data")]
)
def get_channel_user_selection(channels, filename):
    try:
        data_header = read_data_header(filename)
        main_channel_list = data_header["channel_names"]
        main_channel_list = main_channel_list[0]
        user_selected_indexes = [i for i, e in enumerate(main_channel_list) if e in channels]
        user_selected_indexes = [False if i in user_selected_indexes else True for i in range(len(main_channel_list))]
    except:
        user_selected_indexes = [True]
    return json.dumps(channels), user_selected_indexes


# open browser
#chrome_options = Options()
# chrome_options.add_argument("--kiosk")
#driver = webdriver.Chrome(chrome_options=chrome_options)
# driver.get('http://localhost:8050/')

@app.callback(
    [Output({'type': 'ddowns', 'index': ALL}, 'disabled'),
    Output({'type': 'ddowns', 'index': ALL}, 'placeholder'),
    Output({'type': 'ddowns', 'index': ALL}, 'value'),
    Output({'type': 'ddowns', 'index': ALL}, 'style'),
    Output({'type': 'mins', 'index': ALL}, 'disabled'),
    Output({'type': 'mins', 'index': ALL}, 'placeholder'),
    Output({'type': 'mins', 'index': ALL}, 'value'),
    Output({'type': 'mins', 'index': ALL}, 'style'),
    Output({'type': 'maxes', 'index': ALL}, 'disabled'),
    Output({'type': 'maxes', 'index': ALL}, 'placeholder'),
    Output({'type': 'maxes', 'index': ALL}, 'value'),
    Output({'type': 'maxes', 'index': ALL}, 'style'),],
    Input("user-selected-indexes", "data")
)
def toggle_disable(indx):
    try:
        new_placeholders_ddowns = ['N/A' if i else 'Select' for i in indx]
        values_ddowns = ['' if i else dash.no_update for i in indx]
        style_ddowns = [{'width':'110px', 'filter':'blur(150px)', 'transition': 'all 0.5s ease-out', 'opacity':'0'} if i else {'width':'110px', 'filter':'blur(0px)', 'transition': 'all 0.5s ease-in', 'opacity':'100'} for i in indx]
        new_placeholders_mins = ['Min' if i else 'Min' for i in indx] # in case we decide to change placeholders in the future
        values_mins = ['' if i else dash.no_update for i in indx]
        style_mins = [{'width':'80px', 'filter':'blur(150px)', 'transition': 'all 0.5s ease-out', 'opacity':'0'} if i else {'width':'80px', 'filter':'blur(0px)', 'transition': 'all 0.5s ease-in', 'opacity':'100'} for i in indx]
        new_placeholders_maxes = ['Max' if i else 'Max' for i in indx] # in case we decide to change placeholders in the future
        values_maxes = ['' if i else dash.no_update for i in indx]
        style_maxes = [{'width':'80px', 'filter':'blur(150px)', 'transition': 'all 0.5s ease-out', 'opacity':'0'} if i else {'width':'80px', 'filter':'blur(0px)', 'transition': 'all 0.5s ease-in', 'opacity':'100'} for i in indx]
    except:
        new_placeholders_ddowns = ''
        values_ddowns = ''
        new_placeholders_mins = ''
        values_mins = ''
        new_placeholders_maxes = ''
        values_maxes = ''
    return indx, new_placeholders_ddowns, values_ddowns, style_ddowns, indx, new_placeholders_mins, values_mins, style_mins, indx, new_placeholders_maxes, values_maxes, style_maxes


@app.callback(
    Output("save-button", "children"),

    [Input("save-button", "n_clicks"),
     Input("input-file-loc", "data"),
     Input("scoring-labels", "data")]
)
def save_button(n_clicks, input_data_loc, scoring_results):

    if n_clicks:
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


# training
@app.callback(
    [Output("accuracy", "figure"),
     Output("AI-accuracy", "data"),
     Output("table-contents", "children")],

    [Input("AI-trigger-params", "data"),
     Input("AI-accuracy", "data")]
)
def train_indicator(ai_params, ai_acc):
    if (not ai_acc is None) and (not ai_params is None):

        # read params
        ai_params = pd.read_json(ai_params)
        epoch_index = ai_params["epoch_index"][0]
        score_storage = ai_params["score_storage"][0]
        save_path = ai_params["save_path"][0]
        #
        epoch_index = int(epoch_index)

        # change score_storage from json to pandas
        score_storage = pd.read_json(score_storage)

        # check if there is scoring available
        if len(score_storage.values[0]) > 5:

            # check recorded class distribution
            rec_class = np.unique(score_storage.values[0])

            # initializing features vector and labels
            features = []
            labels = []
            # concatinate data
            for epoch in score_storage.keys():
                try:
                    df_mid = []
                    df_mid = pd.read_json(os.path.join(
                        save_path, str(epoch_index) + ".json"))
                    ps_mid = np.stack(df_mid["spectrums"]).T
                    hist_mid = np.stack(df_mid["histograms"]).T

                    # concatinating features
                    features.append(np.vstack([ps_mid, hist_mid]))

                    # get labels and cat them
                    labels.append(score_storage[epoch].values[0])
                except:
                    print(
                        f"Dataset for epoch #{epoch} is not found. Ignoring this epoch for training.")

            # split train test
            features = np.stack(features)

            # get features dimensions
            n, p1, p2 = features.shape

            # reshape features
            features = features.reshape(-1, p1*p2)

            # split
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42)

            # start training (at the moment only using XGBoost)
            start_time = time.time()
            full_study, best_classifier = Classifier(Xtrain=X_train,
                                                     ytrain=y_train,
                                                     Xtest=X_test,
                                                     ytest=y_test).run_xgboost(n_trials=20)
            print(
                f'execution time: {np.rint(time.time() - start_time)} seconds')

            # updating AI-Accuracy vector
            ai_acc = np.hstack(
                [np.squeeze(pd.read_json(ai_acc).values), [full_study.best_value]])

            # get best classifier
            the_classifier = XGBClassifier(
                **full_study.best_trial.params).fit(X_train, y_train)

            # get confusion matrix
            conf_df = get_confusion_mat(y_test, the_classifier.predict(
                X_test), [str(name) for name in np.sort(np.unique(y_test))])

            return get_acc_plot(data=ai_acc), pd.DataFrame({"accuray": ai_acc}).to_json(), conf_df
        else:
            print(
                "Score storage is empty or doesn't satisfy proper epoch number. Training canceled!")
            raise PreventUpdate

    return get_acc_plot(data=np.array([0, 0])), pd.DataFrame({"accuray": [0]}).to_json(), dash.no_update


# run app if it get called
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
