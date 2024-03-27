import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import glasbey
import itertools as it


def color_code_labels(df, label_name="label_embedding", color_noise_black=False, drop_noise=False):
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


def update_heatmap(score, eps, min_samples):
    print("update heatmap", score, eps, min_samples)
    # create new heatmap
    new_field = pd.DataFrame(data[score_map[score]].to_numpy().reshape(len(epss), len(min_sampless)),
                             index=epss, columns=min_sampless)

    # get new current heatmap value
    score_value = new_field.loc[eps, min_samples]

    # update the heatmap figure with a red frame around the selected point
    new_fig = px.imshow(new_field, aspect="auto",
                        labels={"x": "min_samples", "y": "eps", "color": score},
                        color_continuous_scale='gray')

    # update red frame on heatmap element
    new_fig = update_heatmap_selection(fig=new_fig, eps=eps, min_samples=min_samples)

    return new_fig, score_value


def update_heatmap_selection(fig, eps, min_samples):
    # update red frame on heatmap element
    fig["layout"]["shapes"] = [
        go.layout.Shape(
            type='rect',
            x0=min_samples - dx / 2, y0=eps - dy / 2,
            x1=min_samples + dx / 2, y1=eps + dy / 2,
            line=dict(color='red', width=2)
        )]
    fig["layout"]["margin"] = dict(l=margin, r=margin, t=margin, b=margin)
    return fig


def update_geo_and_umap(eps, min_samples, noise_check_value, label_selection=[]):
    print("update geo and umap", eps, min_samples, noise_check_value, label_selection)

    # filter data
    cur_labels = labels[(labels.eps == eps) & (labels.min_samples == min_samples)]
    cur_labels = color_code_labels(cur_labels, label_name=data_label)

    # show or hide noise
    n = "noise"
    if noise_check_value == ["Hide noise"]:
        n = "no_noise"
        cur_labels = cur_labels[cur_labels[data_label] != -1]

    # select specific labels
    if label_selection:
        cur_labels = cur_labels[cur_labels[data_label].isin(label_selection)]

    # define figures
    figure_geo = go.Figure(data=go.Scatter3d(name=f"{eps}-{min_samples}-{n}-geo",
                                             x=cur_labels.LONGITUDE, y=cur_labels.LATITUDE, z=cur_labels.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=cur_labels.color, opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z}<br>' +
                                                           'Label: %{text}<extra></extra>',
                                             text=cur_labels[data_label]
                                             ))
    figure_umap = go.Figure(data=go.Scatter3d(name=f"{eps}-{min_samples}-{n}-umap",
                                              x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                              mode='markers',
                                              marker=dict(size=scatter_size, color=cur_labels.color, opacity=1),
                                              hovertemplate='Longitude: %{x}<br>' +
                                                            'Latitude: %{y}<br>' +
                                                            'Depth: %{z}<br>' +
                                                            'Label: %{text}<extra></extra>',
                                              text=cur_labels[data_label]
                                              ))

    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                             uirevision=True)
    figure_umap.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                              scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                              uirevision=True)

    return figure_geo, figure_umap


def update_text(eps, min_samples, score, score_value):
    print("update text", eps, min_samples, score, score_value)
    if score_value:
        score_value = np.round(score_value, 2)
    return f"Current parameters: \neps={np.round(eps, 8)}\nmin_samples = {min_samples}\n{score} = {score_value}"


def update_depth(depth, eps, min_samples, noise_check_value):
    print("update depth", depth)
    # filter data
    cur_labels = labels[(labels.eps == eps) & (labels.min_samples == min_samples)]
    cur_labels = color_code_labels(cur_labels, label_name=data_label)

    # show or hide noise
    n = "noise"
    if noise_check_value == ["Hide noise"]:
        n = "no_noise"
        cur_labels = cur_labels[cur_labels[data_label] != -1]

    # filter for depth
    temp_labels = cur_labels[cur_labels.LEV_M == cur_labels.LEV_M.unique()[depth]]

    # define figure
    figure_depth = go.Figure(data=go.Scattergeo(name=f"{depth}-{n}-depth", lon=temp_labels.LONGITUDE,
                                                lat=temp_labels.LATITUDE,
                                                mode='markers', marker=dict(color=temp_labels.color),
                                                hovertemplate='Longitude: %{x}<br>' +
                                                              'Latitude: %{y}<br>' +
                                                              'Label: %{text}<extra></extra>',
                                                text=temp_labels[data_label]
                                                ))
    figure_depth.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin), uirevision=True)

    return figure_depth


# plot settings
scatter_size = 2
margin = 5

# which clustering? on original or embedding?
data_label = "label_embedding"
# data_label = "label_original"

# dbscan labels
labels = pd.read_csv("../data/dbscan_labels.csv")

# heatmap data
data = pd.read_csv("../data/dbscan_scores_incomplete.csv")
groupby_cols = ['clustering_on', 'scores_on', 'eps', 'min_samples']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data[(data.clustering_on == data_label.split("_")[1]) & (data.scores_on == data_label.split("_")[1])]  # filter
data = data.drop(['clustering_on', 'scores_on'], axis=1)
data = data.sort_values(["eps", "min_samples"])
data.nnoise = data.nnoise * 100 / 49131
epss = np.sort(data.eps.unique())  # all epsilons
min_sampless = np.sort(data.min_samples.unique())  # all min_samples
all_combos = list(it.product(*[epss, min_sampless]))  # all combinations
score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski", "Davies-Bouldin": "davies_bouldin",
             "N clusters": "nclusters", "N noise": "nnoise"}
dx = abs(min_sampless[0] - min_sampless[1])
dy = abs(epss[0] - epss[1])

print("data loaded")

# current hyperparameter values
cur_eps = 0.01
cur_min_samples = 2
cur_score = "Calinski-Harabasz"
cur_depth = 0  # this is the ID of the depth in the depth list
cur_noise_check_value = ["Hide noise"]

# figures and heatmaps
fig_heatmap, cur_score_value = update_heatmap(score=cur_score, eps=cur_eps, min_samples=cur_min_samples)
fig_geo, fig_umap = update_geo_and_umap(eps=cur_eps, min_samples=cur_min_samples,
                                        noise_check_value=cur_noise_check_value)
fig_depth = update_depth(depth=cur_depth, eps=cur_eps, min_samples=cur_min_samples,
                         noise_check_value=cur_noise_check_value)
cur_text = update_text(eps=cur_eps, min_samples=cur_min_samples, score=cur_score, score_value=cur_score_value)

print("figures defined")

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.RadioItems(id="score", value='Calinski-Harabasz',
                       options=['N clusters', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'N noise'],
                       labelStyle={'display': 'inline-block'}),
        dcc.Checklist(id="hide-noise-check", options=["Hide noise"], value=cur_noise_check_value),
    ]),
    html.Div(
        dcc.Graph(id='heatmap-score', figure=fig_heatmap,
                  clickData={'points': [{'x': cur_min_samples, 'y': cur_eps, 'z': cur_score_value}]}),
        style={'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        dcc.Graph(id='fig-geo', figure=fig_geo),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(id="textarea", children=cur_text,
             style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '49vw'}
             ),
    html.Div(
        dcc.Graph(id='fig-umap', figure=fig_umap),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        dcc.Graph(id="fig-depth", figure=fig_depth, clickData={'points': [{'lon': None, 'lat': None}]}),
        style={'display': 'inline-block', 'width': '49vw'}
    ),
    html.Div(
        [dcc.Slider(id="depth-slider", min=0, max=len(labels.LEV_M.unique()) - 1, step=None, value=cur_depth,
                    marks={i: str(x) for i, x in enumerate(np.sort(labels.LEV_M.unique()))}),
         dcc.RadioItems(id="selection-state", value='select all',
                        options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})],
        style={'display': 'inline-block', 'width': '49vw'}
    ),

    dcc.Store(id="cur-params", data={"eps": cur_eps, "min_samples": cur_min_samples, 'score': cur_score,
                                     "score_value": cur_score_value, "depth": cur_depth,
                                     "selected_labels": [],
                                     'clickData_depth': {'points': [{'lon': None, 'lat': None}]}}),
    dcc.Store(id="rotation", data={"umap_relayout": {}, "geo_relayout": {}, "depth_relayout": {}})
])


@app.callback(
    Output('rotation', "data"),
    Input('rotation', 'data'),
    Input('fig-geo', 'relayoutData'),
    Input('fig-umap', 'relayoutData'),
    Input('fig-depth', 'relayoutData')
)
def update_rotation(rotation, geo_relayout, umap_relayout, depth_relayout):
    # print("update rotation callback")
    new_rotation = rotation.copy()
    new_rotation["umap_relayout"] = umap_relayout
    new_rotation["geo_relayout"] = geo_relayout
    new_rotation["depth_relayout"] = depth_relayout

    return new_rotation


@app.callback(
    Output('heatmap-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('fig-depth', 'figure'),
    Output('textarea', 'children'),
    Output('cur-params', 'data'),

    Input('heatmap-score', 'figure'),
    Input('heatmap-score', 'clickData'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('fig-depth', 'figure'),
    Input('fig-depth', 'clickData'),  # selectedData
    Input('score', 'value'),
    Input('hide-noise-check', 'value'),
    Input('depth-slider', 'value'),
    Input('cur-params', 'data'),
    Input('selection-state', 'value'),
    State('rotation', 'data')
)
def update(figure_heatmap, clickData_heatmap, figure_geo, figure_umap, figure_depth, clickData_depth,
           new_score, check_value, new_depth, cur_params, selection_state, cur_rotation):
    # print("update callback")
    # get data from previous state
    prev_eps = cur_params["eps"]
    prev_min_samples = cur_params["min_samples"]
    prev_score = cur_params["score"]
    prev_depth = cur_params["depth"]
    prev_score_value = cur_params["score_value"]
    prev_selected_labels = cur_params["selected_labels"]
    prev_clickData_depth = cur_params["clickData_depth"]

    # get data from new state
    new_min_samples = clickData_heatmap['points'][0]['x']
    new_eps = clickData_heatmap['points'][0]['y']

    # init new state
    new_heatmap_fig = figure_heatmap
    new_geo_fig = figure_geo
    new_umap_fig = figure_umap
    new_depth_fig = figure_depth
    new_txt = update_text(prev_eps, prev_min_samples, prev_score, prev_score_value)
    new_params = {"eps": prev_eps, "min_samples": prev_min_samples, "score": prev_score,
                  "score_value": prev_score_value, "depth": prev_depth, "selected_labels": prev_selected_labels,
                  "clickData_depth": prev_clickData_depth}

    # if score is new, redraw heatmap
    if new_score != prev_score:
        print("new score")
        new_heatmap_fig, new_score_value = update_heatmap(score=new_score, eps=new_eps, min_samples=new_min_samples)
        new_txt = update_text(eps=new_eps, min_samples=new_min_samples, score=new_score, score_value=new_score_value)

        new_params["score"] = new_score
        new_params["score_value"] = new_score_value

    # if eps or min_samples are new, update geo, umap and depth figures
    if prev_eps != new_eps or prev_min_samples != new_min_samples:
        print("new eps or min_samples")
        # update heatmap rectangle
        new_heatmap_fig = update_heatmap_selection(fig=new_heatmap_fig, eps=new_eps, min_samples=new_min_samples)

        # find new score value
        min_samples_idx = np.where(np.array(new_heatmap_fig["data"][0]["x"]) == new_min_samples)
        eps_idx = np.where(np.array(new_heatmap_fig["data"][0]["y"]) == new_eps)
        new_score_value = np.array(new_heatmap_fig["data"][0]["z"])[eps_idx, min_samples_idx][0, 0]

        # update label plots
        new_geo_fig, new_umap_fig = update_geo_and_umap(eps=new_eps, min_samples=new_min_samples,
                                                        noise_check_value=check_value,
                                                        label_selection=[])
        new_depth_fig = update_depth(depth=prev_depth, eps=new_eps, min_samples=new_min_samples,
                                     noise_check_value=check_value)
        new_txt = update_text(eps=new_eps, min_samples=new_min_samples, score=new_score,
                              score_value=new_score_value)

        new_params["eps"] = new_eps
        new_params["min_samples"] = new_min_samples
        new_params["score_value"] = new_score_value
        new_params["selected_labels"] = []

    # if depth slider changed, update depth figure
    if new_depth != prev_depth:
        print(f"new_depth {new_depth}, prev_depth {prev_depth}")
        new_depth_fig = update_depth(depth=new_depth, eps=prev_eps, min_samples=prev_min_samples,
                                     noise_check_value=check_value)

        new_params["depth"] = new_depth

    # if a click happened in the depth label plot, show that specific cluster only (select and deselect?)
    if clickData_depth['points'][0]['lon']:
        print("clickData depth", clickData_depth)
        # find the label of the clicked point
        lat = clickData_depth['points'][0]['lat']
        lon = clickData_depth['points'][0]['lon']

        cur_labels = labels[(labels.eps == new_eps) & (labels.min_samples == new_min_samples)]
        selected_label = cur_labels[(cur_labels.LATITUDE == lat) &
                                    (cur_labels.LONGITUDE == lon) &
                                    (cur_labels.LEV_M == labels.LEV_M.unique()[new_depth])][data_label]
        if selected_label.empty:
            selected_label = []
        else:
            selected_label = [selected_label.values[0]]

        print(lat, lon, prev_selected_labels)

        # only update figures if the click data is different to the previous click data7
        new_selected_labels = prev_selected_labels
        if selection_state == "select all":
            new_selected_labels = []

        if prev_clickData_depth != clickData_depth:
            if selection_state == "select":
                new_selected_labels = prev_selected_labels + selected_label
            elif selection_state == "deselect":
                new_selected_labels = [x for x in prev_selected_labels if x != selected_label[0]]

        print(lat, lon, new_selected_labels)

        # update label selection
        new_params["selected_labels"] = new_selected_labels
        new_params["clickData_depth"] = clickData_depth

        # update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(eps=new_eps, min_samples=new_min_samples,
                                                        noise_check_value=check_value,
                                                        label_selection=new_selected_labels)

    # apply previous rotation
    if cur_rotation:
        print(figure_depth)
        print(cur_rotation)
        if "umap_relayout" in cur_rotation.keys():
            if cur_rotation["umap_relayout"]:
                if "scene.camera" in cur_rotation["umap_relayout"].keys():
                    print("umap relayout")
                    new_umap_fig["layout"]["scene.camera"] = cur_rotation["umap_relayout"]["scene.camera"]
        if "geo_relayout" in cur_rotation.keys():
            if cur_rotation["geo_relayout"]:
                if "scene.camera" in cur_rotation["geo_relayout"].keys():
                    print("geo relayout")
                    new_geo_fig["layout"]["scene.camera"] = cur_rotation["geo_relayout"]["scene.camera"]
        if "depth_relayout" in cur_rotation.keys():
            if cur_rotation["depth_relayout"]:
                if "scene.camera" in cur_rotation["depth_relayout"].keys():
                    print("depth relayout")
                    new_depth_fig["layout.geo.projection.rotation.lon"] = cur_rotation["depth_relayout"]["projection.rotation.lon"]

    return new_heatmap_fig, new_geo_fig, new_umap_fig, new_depth_fig, new_txt, new_params


# import pandas as pd
# import numpy as np
# data_label = "label_embedding"
# # data_label = "label_original"
# labels = pd.read_csv("data/dbscan_labels.csv")
# df_in = pd.read_csv("data/df_wide_knn.csv")
def get_info(cluster_label, labels, df_in, eps=0.10983051, min_samples=4):
    # filter for the correct hyperparameter combination
    temp = labels[(np.round(labels.eps, 8) == np.round(eps, 8)) & (labels.min_samples == min_samples)]
    all_labels = temp[data_label].unique()  # find all labels of that clustering

    # check if cluster label exists
    if cluster_label in all_labels:
        temp = temp[temp[data_label] == cluster_label]  # filter out the cluster label
        # merge labels to parameter information
        temp_merged = pd.merge(left=temp, right=df_in, how="left", on=["LATITUDE", "LONGITUDE", "LEV_M"])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(f"Information on cluster label {cluster_label}")
            print(temp_merged[[x for x in temp_merged.columns
                               if x not in ["eps", "min_samples", "label_embedding", "label_original"]]].describe())

        return temp_merged
    else:
        print(f"Cluster label {cluster_label} not found.")
        return


def difference_between_two_labels(a, b, labels, df_in, eps=0.10983051, min_samples=4):
    a_data = get_info(a, labels, df_in, eps, min_samples)
    b_data = get_info(b, labels, df_in, eps, min_samples)

    diff = (a_data.describe() - b_data.describe())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(f"Information on the difference of cluster labels {a} and {b}")
        print(diff[[x for x in a_data.columns if x not in ["eps", "min_samples", "label_embedding", "label_original"]]])

    # return diff


# run app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
