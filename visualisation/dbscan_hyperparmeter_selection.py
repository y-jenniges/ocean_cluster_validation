# no traces but data loading
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


def compute_clustering(df, eps, min_samples, old_params, data_label="embedding"):
    print("compute clustering")
    temp = df.copy()

    # compute embedding (only if parameters changed)
    if old_params["eps"] == eps and old_params["min_samples"] == min_samples:
        print("recompute embedding")
        new_embedding = umap.UMAP(min_dist=min_dist, n_components=n_components,
                              n_neighbors=n_neighbors).fit_transform(df_scaled)
        temp["e0"] = new_embedding[:, 0]
        temp["e1"] = new_embedding[:, 1]
        temp["e2"] = new_embedding[:, 2]

        # compute clustering
        if data_label == "embedding":
            cur_model = DBSCAN(eps=eps, min_samples=min_samples).fit(new_embedding)
        else:
            cur_model = DBSCAN(eps=eps, min_samples=min_samples).fit(df_scaled)
    else:
        # compute clustering
        if data_label == "embedding":
            cur_model = DBSCAN(eps=eps, min_samples=min_samples).fit(embedding)
        else:
            cur_model = DBSCAN(eps=eps, min_samples=min_samples).fit(df_scaled)

    temp["label"] = cur_model.labels_
    temp = color_code_labels(temp)

    return temp


def update_geo_and_umap(eps, min_samples, old_params, hide_noise=True, data_label="embedding", label_selection=[]):
    print("update geo and umap", eps, min_samples, hide_noise, label_selection)

    # compute clustering
    df_cur = compute_clustering(df=df_in, eps=eps, min_samples=min_samples, old_params=old_params, data_label=data_label)

    # show or hide noise
    n = "show_noise"
    if hide_noise:
        n = "hide_noise"
        df_cur = df_cur[df_cur["label"] != -1]

    # select specific labels
    if label_selection:
        df_cur = df_cur[df_cur["label"].isin(label_selection)]

    # define figures
    figure_geo = go.Figure(data=go.Scatter3d(#name=f"{eps}-{min_samples}-{n}-geo",
                                             x=df_cur.LONGITUDE, y=df_cur.LATITUDE, z=df_cur.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df_cur.color, opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z}<br>' +
                                                           'Label: %{text}<extra></extra>',
                                             text=df_cur["label"]
                                             ))
    figure_umap = go.Figure(data=go.Scatter3d(#name=f"{eps}-{min_samples}-{n}-umap",
                                              x=df_cur.e0, y=df_cur.e1, z=df_cur.e2,
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

# initial hyperparameters and their range
eps = 0.01
min_samples = 3
hyp_range = {"eps": np.linspace(0.01, 0.2, 60), "min_samples": range(2, 12)}

# plot settings
scatter_size = 2
margin = 5

# init plots
fig_geo, fig_umap = update_geo_and_umap(eps, min_samples, hide_noise=True, label_selection=[],
                                        data_label="embedding", old_params=dict(eps=eps, min_samples=min_samples))

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='fig-geo', figure=fig_geo),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '40vw'}
    ),
    html.Div(
        dcc.Graph(id='fig-umap', figure=fig_umap),
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '40vw'}
    ),
    # html.Div([
    #     html.Div("iteration"),
    #     dcc.Dropdown(list(hyp_range["eps"]), value=eps, id='eps-dropdown', clearable=False)
    # ]),
    html.Div([
        html.Div([
            html.Div("eps"),
            dcc.Dropdown(list(hyp_range["eps"]), value=eps, id='eps-dropdown', clearable=False)],
            ),
        html.Div([
            html.Div("min_samples"),
            dcc.Dropdown(list(hyp_range["min_samples"]), value=min_samples, id='min_samples-dropdown', clearable=False)
        ]),
        html.Div([
            html.Div("data"),
            dcc.Dropdown(["original", "embedding"], value="embedding", id='data-dropdown', clearable=False),
        ]),
        html.Button('Compute', id='compute-button', n_clicks=0)
        ],
        style={'margin': dict(l=margin, r=margin, t=margin, b=margin), 'display': 'inline-block', 'width': '10vw'}
    ),
    dcc.Store(id="current", data={"eps": eps, "min_samples": min_samples}),
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
    State('eps-dropdown', 'value'),
    State('min_samples-dropdown', 'value'),
    State('data-dropdown', 'value'),
    State('current', 'data'),
    Input('compute-button', 'n_clicks')
)
def update(eps, min_samples, data_label, old_params, btn_clicks):
    print(btn_clicks)
    figure_geo, figure_umap = update_geo_and_umap(eps, min_samples, hide_noise=True, data_label=data_label,
                                                  label_selection=[], old_params=old_params)
    new_params = dict(eps=eps, min_samples=min_samples)
    print("figures updated")
    return figure_geo, figure_umap, new_params


# run app
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
