# no traces but data loading
import dash.exceptions
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import matplotlib
import glasbey
import itertools as it
import utils


def update_geo_and_umap(data_label="label", hide_noise=True, label_selection=[]):
    print("update geo and umap", label_selection)
    df_display = df.copy()

    # determine labels to display
    if label_selection:
        df_display = df_display[df_display[data_label].isin(label_selection)]

    # hide noise
    if hide_noise:
        df_display = df_display[df_display[data_label] != -1]

    # define figures
    figure_geo = go.Figure(data=go.Scatter3d(x=df_display.LONGITUDE, y=df_display.LATITUDE, z=df_display.LEV_M * -1,
                                             mode='markers',
                                             marker=dict(size=scatter_size, color=df_display.color, opacity=1),
                                             hovertemplate='Longitude: %{x}<br>' +
                                                           'Latitude: %{y}<br>' +
                                                           'Depth: %{z} m<br>' +
                                                           'Temperature: %{text[0]:.2f} °C<br>' +
                                                           'Salinity: %{text[1]:.2f} psu<br>' +
                                                           'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                           'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                           'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                           'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                           'Label: %{text[6]}<extra></extra>',
                                             text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                              "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                              data_label]]
                                             ))
    figure_umap = go.Figure(data=go.Scatter3d(x=df_display.e0, y=df_display.e1, z=df_display.e2,
                                              mode='markers',
                                              marker=dict(size=scatter_size, color=df_display.color, opacity=1),
                                              hovertemplate='X: %{x:.2f}<br>' +
                                                            'Y: %{y:.2f}<br>' +
                                                            'Z: %{z:.2f}<br>' +
                                                            'Temperature: %{text[0]:.2f} °C<br>' +
                                                            'Salinity: %{text[1]:.2f} psu<br>' +
                                                            'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                            'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                            'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                            'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                            'Label: %{text[6]}<extra></extra>',
                                              text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                               "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                               data_label]]
                                              ))

    figure_geo.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                             scene=dict(xaxis_title="Longitude", yaxis_title="Latitude", zaxis_title="Depth [m]"),
                             uirevision=True)
    figure_umap.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin),
                              scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                              uirevision=True)

    return figure_geo, figure_umap


def update_depth(depth_idx, hide_noise=True, data_label="label"):
    print("update depth", depth_idx)
    df_display = df.copy()

    # hide noise
    if hide_noise:
        df_display = df_display[df_display[data_label] != -1]

    # filter for depth
    df_display = df_display[df_display.LEV_M == depths[depth_idx]]

    # define figure
    figure_depth = go.Figure(data=go.Scattergeo(lon=df_display.LONGITUDE,
                                                lat=df_display.LATITUDE,
                                                mode='markers', marker=dict(color=df_display.color),
                                                hovertemplate='Longitude: %{lon}<br>' +
                                                              'Latitude: %{lat}<br>' +
                                                              'Temperature: %{text[0]:.2f} °C<br>' +
                                                              'Salinity: %{text[1]:.2f} psu<br>' +
                                                              'Oxygen: %{text[2]:.2f} µmol/kg<br>' +
                                                              'Nitrate: %{text[3]:.2f} µmol/kg<br>' +
                                                              'Silicate: %{text[4]:.2f} µmol/kg<br>' +
                                                              'Phosphate: %{text[5]:.2f} µmol/kg<br>' +
                                                              'Label: %{text[6]}<extra></extra>',
                                                text=df_display[["P_TEMPERATURE", "P_SALINITY", "P_OXYGEN",
                                                                 "P_NITRATE", "P_SILICATE", "P_PHOSPHATE",
                                                                 data_label]]
                                                ))
    figure_depth.update_layout(margin=dict(l=margin, r=margin, t=margin, b=margin), uirevision=True)
    figure_depth.update_geos(
        lonaxis_range=[round(df.LONGITUDE.min()) - margin, round(df.LONGITUDE.max()) + margin],
        lataxis_range=[round(df.LATITUDE.min()) - margin, round(df.LATITUDE.max()) + margin])
    return figure_depth


# plot settings
scatter_size = 2
margin = 5

# load file to visualise
# df = pd.read_csv("../output_final/dbscan/uncertainty/umap_dbscan_7.csv")
# df = utils.color_code_labels(df)
df = pd.read_csv("../output_final/dbscan/post_processing/re-assigned_A1.csv")
data_label = "label"

# df = pd.read_csv("../output_final/dbscan/uncertainty/uncertainty.csv")
# df.uncertainty = round(df.uncertainty)
# cmap = matplotlib.cm.get_cmap("viridis")
# color_map = {label: matplotlib.colors.rgb2hex(cmap(label/100)) for label in np.sort(df['uncertainty'].unique())}
# df["color"] = df["uncertainty"].map(color_map)
# data_label = "uncertainty"

# define depths
depths = np.sort(df.LEV_M.unique())
cur_depth_idx = 0

# figures
fig_geo, fig_umap = update_geo_and_umap(data_label=data_label, hide_noise=True, label_selection=[])
fig_depth = update_depth(depth_idx=cur_depth_idx, data_label=data_label, hide_noise=True)

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div(
        dcc.Graph(id='fig-geo', figure=fig_geo),
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
        [dcc.Slider(id="depth-slider", min=0, max=len(depths) - 1, step=None, value=cur_depth_idx,
                    marks={i: str(x) for i, x in enumerate(depths)}, vertical=True),
         dcc.RadioItems(id="selection-state", value='select all',
                        options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})],
        style={'display': 'inline-block', 'width': '49vw'}
    ),
    dcc.Store(id="cur-params", data={"depth_idx": cur_depth_idx, "selected_labels": [],
                                     "clickDataSel_depth": {},
                                     'clickData_depth': {'points': [{'lon': None, 'lat': None}]}}
              )
])


@app.callback(
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('fig-depth', 'figure'),
    Output('cur-params', 'data'),

    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('fig-depth', 'figure'),
    Input('fig-depth', 'clickData'),
    Input('fig-depth', 'selectedData'),
    Input('depth-slider', 'value'),
    Input('selection-state', 'value'),
    Input('cur-params', 'data')
)
def update(figure_geo, figure_umap, figure_depth, clickData_depth, clickDataSel_depth, new_depth_idx, selection_state, cur_params):
    # print("update callback")
    # get data from previous state
    prev_depth_idx = cur_params["depth_idx"]
    prev_selected_labels = cur_params["selected_labels"]
    prev_clickData_depth = cur_params["clickData_depth"]
    prev_clickDataSel_depth = cur_params["clickDataSel_depth"]

    # init new state
    new_geo_fig = figure_geo
    new_umap_fig = figure_umap
    new_depth_fig = figure_depth
    new_params = {"depth_idx": prev_depth_idx,
                  "selected_labels": prev_selected_labels,
                  "clickData_depth": prev_clickData_depth,
                  "clickDataSel_depth": prev_clickDataSel_depth}

    # if depth slider changed, update depth figure
    if new_depth_idx != prev_depth_idx:
        print(f"new_depth {new_depth_idx}, prev_depth {prev_depth_idx}")
        new_depth_fig = update_depth(depth_idx=new_depth_idx, data_label=data_label, hide_noise=True)
        new_params["depth_idx"] = new_depth_idx

    # if a click happened in the depth label plot, show that specific cluster only (select and deselect?)
    if clickDataSel_depth:
        print("clickDataSel depth", clickDataSel_depth)
        if selection_state == "select all":
            new_selected_labels = []
        else:
            new_selected_labels = list(set([e["text"] for e in clickDataSel_depth["points"]]))
        new_params["selected_labels"] = new_selected_labels
        new_params["clickDataSel_depth"] = clickDataSel_depth

        # update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, data_label=data_label)

        print("updated geo and umap")

    elif clickData_depth['points'][0]['lon']:
        print("clickData depth", clickData_depth)

        # find the label of the clicked point
        lat = clickData_depth['points'][0]['lat']
        lon = clickData_depth['points'][0]['lon']

        selected_label = df[(df.LATITUDE == lat) &
                            (df.LONGITUDE == lon) &
                            (df.LEV_M == depths[new_depth_idx])][data_label]
        if selected_label.empty:
            selected_label = []
        else:
            selected_label = [selected_label.values[0]]

        print(lat, lon, prev_selected_labels)

        # only update figures if the click data is different to the previous click data
        new_selected_labels = prev_selected_labels
        if selection_state == "select all":
            new_selected_labels = []

        if prev_clickData_depth != clickData_depth:
            if selection_state == "select":
                new_selected_labels = prev_selected_labels + selected_label
            elif selection_state == "deselect":
                new_selected_labels = [x for x in prev_selected_labels if x != selected_label[0]]

        # print(lat, lon, new_selected_labels)
        print(new_selected_labels)

        # update label selection
        new_params["selected_labels"] = new_selected_labels
        new_params["clickData_depth"] = clickData_depth

        # update geo and umap plot accordingly
        new_geo_fig, new_umap_fig = update_geo_and_umap(label_selection=new_selected_labels, data_label=data_label)

    return new_geo_fig, new_umap_fig, new_depth_fig, new_params


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
