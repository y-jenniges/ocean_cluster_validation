import dash.exceptions
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import glasbey
import itertools as it
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import umap
import glob


def color_code_labels(df, label_name="label", color_noise_black=False, drop_noise=False):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # define colors
    unique_labels = np.sort(np.unique(temp[label_name]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp[label_name].map(lambda x: color_map[x])

    # how to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp[label_name] == -1, "color"] = "#000000"
    if drop_noise:
        temp = temp[temp[label_name] != -1]

    return temp


def update_geo_and_umap(iteration, hide_noise=True, label_selection=[], labels_above_thresh=[]):
    print("update geo and umap", iteration, hide_noise, label_selection)

    # compute clustering
    df_cur = df_dbscan[iteration]
    df_cur = color_code_labels(df_cur)

    # show or hide noise
    n = "show_noise"
    if hide_noise:
        n = "hide_noise"
        df_cur = df_cur[df_cur["label"] != -1]

    # only show labels above selected thresh for the count of grid cells
    if labels_above_thresh:
        df_cur = df_cur[df_cur["label"].isin(labels_above_thresh)]

    # select specific labels
    df_cur_selection = df_cur.copy()
    if label_selection:
        df_cur_selection = df_cur[df_cur["label"].isin(label_selection)]

    # define figures
    figure_geo = go.Figure(data=go.Scatter3d(x=df_cur_selection.LONGITUDE, y=df_cur_selection.LATITUDE,
                                             z=df_cur_selection.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df_cur_selection.color, opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z}<br>' +
                                                           'Label: %{text}<extra></extra>',
                                             text=df_cur_selection["label"]
                                             ))
    figure_umap = go.Figure(data=go.Scatter3d(x=df_cur.e0, y=df_cur.e1, z=df_cur.e2,
                                              mode='markers',
                                              marker=dict(size=scatter_size, color=df_cur.color, opacity=1),
                                              hovertemplate='Longitude: %{x}<br>' +
                                                            'Latitude: %{y}<br>' +
                                                            'Depth: %{z}<br>' +
                                                            'Label: %{text}<extra></extra>',
                                              text=df_cur["label"]
                                              ))

    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                             uirevision=True)
    figure_umap.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                              scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                              uirevision=True)

    return figure_geo, figure_umap


def update_elbow(iteration, thresh=None, label=None, y_scale="log"):
    # get current clustering
    df_cur = df_dbscan[iteration]

    # count number of grid cells in each cluster
    some_column = df_cur.columns[0]
    df_nums = df_cur.groupby("label").count()[some_column].reset_index().rename(columns={some_column: "count"})
    df_nums = df_nums.sort_values("count").reset_index(drop=True)
    df_nums["label"] = df_nums["label"].astype(str)

    # define figure
    figure_elbow = go.Figure(data=go.Scatter(x=df_nums["label"], y=df_nums["count"], mode='lines+markers',
                                             # marker=dict(size=scatter_size, color=df_cur_selection.color, opacity=1),
                                             hovertemplate='Label: %{x}<br>' +
                                                           'Count: %{y}<br>'
                                             ))
    figure_elbow.update_yaxes(type=y_scale)
    figure_elbow.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                               scene=dict(xaxis_title="Label", yaxis_title="Grid cell count"),
                               uirevision=True)
    labels_above_thresh = []
    print(f"thresh {thresh}")
    if thresh:
        labels_above_thresh = list(df_nums[df_nums["count"] > thresh].label.astype(int))
        if not label:
            x = df_nums[df_nums["count"] == thresh]["label"].index[0]
        else:
            x = df_nums[df_nums["label"] == label]["label"].index[0]
        figure_elbow.add_hline(y=thresh)
        figure_elbow.add_vline(x=x)

    return figure_elbow, labels_above_thresh


# load pre-computed DBSCAN files
num_iterations = 100
df_dbscan = []
for i in range(num_iterations):
    df_dbscan.append(pd.read_csv(f"C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/"
                                 f"uncertainty/umap_dbscan_{i}.csv"))

# load data
df_in = pd.read_csv("../data/df_wide_knn.csv")
df_params = df_in.drop(["LATITUDE", "LONGITUDE", "LEV_M"], axis=1)  # remove geolocation

# scale data
scaler = MinMaxScaler().fit(df_params)
df_scaled = pd.DataFrame(scaler.transform(df_params), columns=df_params.columns)

# compute embedding
min_dist = 0.0
n_components = 3
n_neighbors = 20
embedding = umap.UMAP(min_dist=min_dist, n_components=n_components,
                      n_neighbors=n_neighbors).fit_transform(df_scaled)
df_in["e0"] = embedding[:, 0]
df_in["e1"] = embedding[:, 1]
df_in["e2"] = embedding[:, 2]

# plot settings
scatter_size = 2
margin = 5

# init plots
iteration = 0
fig_geo, fig_umap = update_geo_and_umap(iteration, hide_noise=True, label_selection=[])
fig_elbow, labels_above_thresh = update_elbow(iteration, thresh=None)

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='fig-geo', figure=fig_geo),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '40vw'}
    ),
    html.Div(
        dcc.Graph(id='fig-umap', figure=fig_umap,
                  clickData={'points': [{'x': None, 'y': None, 'z': None, 'text': None}]}),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '40vw'}
    ),
    html.Div([
        html.Div("iteration"),
        dcc.Slider(min=0, max=99, step=1, value=0, id='iteration-dropdown')
    ]),
    html.Div(
        dcc.RadioItems(id="selection-state", value='select all',
                       options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})
    ),
    html.Div([
        dcc.Graph(id='fig-elbow', figure=fig_elbow, clickData={'points': [{'x': None, 'y': None, 'text': None}]}),
        dcc.RadioItems(id="y-scale", value='log', options=['log', 'linear'], labelStyle={'display': 'inline-block'}),
        html.Button('Clear thresh', id='clear-btn', n_clicks=0),
        html.Button('Automatic thresh', id='auto-btn', n_clicks=0),
        html.Div("Drop all clusters with less than ... cells.")
    ]),
    dcc.Store(id="current", data={"label_selection": [], "umap_clickData": None, "elbow_click": None,
                                  "iteration": iteration, "labels_above_thresh": [], "y_scale": "log", "thresh": None,
                                  "label": None}),
    dcc.Store(id="rotation", data={"geo_relayout": {}, "umap_relayout": {}})
])


@app.callback(
    Output('rotation', 'data'),
    Input('fig-geo', 'relayoutData'),
    Input('fig-umap', 'relayoutData')
)
def update_rotation(geo_relayout, umap_relayout):
    new_rotation = dict()
    new_rotation["umap_relayout"] = umap_relayout
    new_rotation["geo_relayout"] = geo_relayout
    return new_rotation


@app.callback(
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('current', 'data'),
    Output("fig-elbow", "figure"),
    Input('iteration-dropdown', 'value'),
    Input('fig-umap', 'clickData'),
    Input('selection-state', 'value'),
    Input('current', 'data'),
    Input("fig-elbow", "clickData"),
    Input("y-scale", "value"),
    Input("fig-elbow", "figure")
)
def update(iteration, umap_click, selection_state, old_params, elbow_click, y_scale, figure_elbow):
    print(iteration)
    print(f"elbow click: {elbow_click}")
    print(f"umap click: {umap_click}")
    print("old_params", old_params)

    prev_label_selection = old_params["label_selection"]
    prev_umap_clickData = old_params["umap_clickData"]
    prev_elbow_clickData = old_params["elbow_click"]
    prev_iteration = old_params["iteration"]
    prev_labels_above_thresh = old_params["labels_above_thresh"]
    prev_y_scale = old_params["y_scale"]
    prev_thresh = old_params["thresh"]
    prev_label = old_params["label"]

    # check if there is a thresh for grid cell numbers per cluster
    thresh = prev_thresh
    label = prev_label
    labels_above_thresh = prev_labels_above_thresh
    if elbow_click and (prev_elbow_clickData != elbow_click):
        print("elbow click new")
        label = elbow_click["points"][0]["x"]
        count = elbow_click["points"][0]["y"]
        print(label, count)
        thresh = count

        figure_elbow, labels_above_thresh = update_elbow(iteration=iteration, thresh=thresh, label=label,
                                                         y_scale=y_scale)

    # check if iteration or y_scale changed and update elbow plot
    if iteration != prev_iteration or y_scale != prev_y_scale:
        figure_elbow, labels_above_thresh = update_elbow(iteration=iteration, thresh=thresh, label=label,
                                                         y_scale=y_scale)

    new_label_selection = []
    if selection_state == 'select':
        if umap_click and (prev_umap_clickData != umap_click):
            print("select")
            selected_label = umap_click["points"][0]["text"]
            new_label_selection = prev_label_selection + [selected_label]
    elif selection_state == 'deselect':
        if umap_click and (prev_umap_clickData != umap_click):
            print("deselect")
            selected_label = umap_click["points"][0]["text"]
            new_label_selection = [x for x in prev_label_selection if x != selected_label]
    print("New label selection: ", new_label_selection)

    figure_geo, figure_umap = update_geo_and_umap(iteration=iteration, hide_noise=True,
                                                  label_selection=new_label_selection,
                                                  labels_above_thresh=labels_above_thresh)

    new_params = dict(label_selection=new_label_selection, umap_clickData=umap_click, elbow_click=elbow_click,
                      iteration=iteration, labels_above_thresh=labels_above_thresh, y_scale=y_scale, thresh=thresh,
                      label=label)

    print("figures updated")
    print()
    return figure_geo, figure_umap, new_params, figure_elbow


# run app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)

# option automatic thresh
