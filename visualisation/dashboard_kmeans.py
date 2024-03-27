import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import glasbey


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


# plot settings
scatter_size = 1.5

# which clustering? on original or embedding?
data_label = "label_embedding"
# data_label = "label_original"

# dbscan labels
labels = pd.read_csv("../data/kmeans_labels.csv")

# heatmap data
data = pd.read_csv("../data/kmeans_scores.csv")
data = data.drop(data[data.n_clusters == 1].index).reset_index(drop=True)
groupby_cols = ['clustering_on', 'scores_on', 'n_clusters', 'n_init']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data[(data.clustering_on == data_label.split("_")[1]) & (data.scores_on == data_label.split("_")[1])]  # filter
data = data.drop(['clustering_on', 'scores_on'], axis=1)
data = data.sort_values(['n_clusters', 'n_init'])
n_clusterss = np.sort(data.n_clusters.unique())  # all n_clusters

score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski", "Davies-Bouldin": "davies_bouldin",
             "N clusters": "nclusters"}

print("data loaded")

# current heatmap data
cur_score = "calinski"

# figures and heatmaps
fig_geo = go.Figure()
fig_umap = go.Figure()
# fig_depth = go.Figure()
line_score = px.line(data, x='n_clusters', y=cur_score, markers=True)  # , aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": cur_score}, color_continuous_scale='gray')
# line_score.update_traces(marker=dict(color=['red']*len(n_clusterss)))

# current hyperparameter values
cur_n_clusters = 2

print("figures defined")

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div([dcc.RadioItems(['N clusters', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette'],
                             'Calinski-Harabasz',
                             id='score', labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
              ]),
    html.Div(dcc.Graph(figure=line_score, id='line-score', clickData={'points': [
                           {'x': cur_n_clusters, 'y': data[data.n_clusters == cur_n_clusters].iloc[0][cur_score], 'pointNumber':  0}]}
                       ),
             style={'display': 'inline-block', 'width': '49%'}),
    html.Div(dcc.Graph(figure=fig_geo, id='fig-geo'),
             style={'margin': dict(l=20, r=20, t=20, b=20),  # "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),
    html.Div(
        dcc.RadioItems(id="selection-state", value='select all',
                       options=['select all', 'select', 'deselect'], labelStyle={'display': 'inline-block'})
    ),
    html.Div([dcc.Textarea(id="textarea", value="Current parameters: "),
              # dcc.Graph(figure=fig_depth, id="fig-depth"),
              # dcc.Slider(0, len(labels.LEV_M.unique())-1, value=0, id='depth-slider')
              ],
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%",
                    "height": "49%", "resize": "none", 'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap',
                       clickData={'points': [{'x': None, 'y': None, 'z': None, 'text': None}]}),
             style={'margin': dict(l=20, r=20, t=20, b=20),  # "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),
    dcc.Store(id="current", data={"label_selection": [], "umap_clickData": None}),
])


@app.callback(
    Output('textarea', 'value'),
    Output('line-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('current', 'data'),

    Input('score', 'value'),
    Input('line-score', 'clickData'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
    Input('fig-umap', 'clickData'),
    Input('selection-state', 'value'),
    Input('current', 'data')
)
def update_heatmap(score, clickData, figure_geo, figure_umap, umap_clickData, selection_state, old_params):
    # update heatmap
    new_line = px.line(data, x='n_clusters', y=score_map[score], markers=True)
    new_line.update_traces(marker=dict(color=['red'] * len(n_clusterss)))

    # update label selection
    prev_label_selection = old_params["label_selection"]
    prev_umap_clickData = old_params["umap_clickData"]
    new_label_selection = []
    if selection_state == 'select':
        if umap_clickData and (prev_umap_clickData != umap_clickData):
            print("select")
            selected_label = umap_clickData["points"][0]["text"]
            new_label_selection = prev_label_selection + [selected_label]
    elif selection_state == 'deselect':
        if umap_clickData and (prev_umap_clickData != umap_clickData):
            print("deselect")
            selected_label = umap_clickData["points"][0]["text"]
            new_label_selection = [x for x in prev_label_selection if x != selected_label]
    print(new_label_selection)

    if clickData:
        # get click coordinates
        x = clickData['points'][0]['x']
        score_value = data[data.n_clusters == x][score_map[score]].values[0]
        point_number = clickData['points'][0]['pointNumber']

        # draw selected point red, all others blue
        trace = next(new_line.select_traces())
        colors = ['blue'] * len(trace.x)
        colors[point_number] = 'red'
        trace.marker.color = colors

        # update label plots
        cur_labels = labels[labels.n_clusters == x]
        cur_labels = color_code_labels(cur_labels, label_name=data_label)

        geo_labels = cur_labels.copy()
        if new_label_selection:
            geo_labels = geo_labels[cur_labels[data_label].isin(new_label_selection)]

        figure_geo = go.Figure(data=go.Scatter3d(name=f"{x}-geo",
                                                 x=geo_labels.LONGITUDE, y=geo_labels.LATITUDE, z=geo_labels.LEV_M * -1,
                                                 mode='markers',
                                                 marker=dict(size=scatter_size, color=geo_labels.color, opacity=1),
                                                 hovertemplate='Longitude: %{x}<br>' +
                                                               'Latitude: %{y}<br>' +
                                                               'Depth: %{z}<br>' +
                                                               'Label: %{text}<extra></extra>',
                                                 text=geo_labels[data_label]
                                                 ))
        figure_umap = go.Figure(data=go.Scatter3d(name=f"{x}-umap",
                                                  x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                                  mode='markers',
                                                  marker=dict(size=scatter_size, color=cur_labels.color, opacity=1),
                                                  hovertemplate='x: %{x}<br>' +
                                                                'y: %{y}<br>' +
                                                                'z: %{z}<br>' +
                                                                'Label: %{text}<extra></extra>',
                                                  text=cur_labels[data_label]
                                                  ))
        figure_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # , paper_bgcolor="LightSteelBlue")
        figure_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # , paper_bgcolor="LightSteelBlue")

        return 'Current parameters: \nn_clusters = {}\n{} = {}'.format(x, score_map[score], np.round(score_value, 2)), \
               new_line, figure_geo, figure_umap, \
               {"label_selection": new_label_selection, "umap_clickData": umap_clickData}
    else:
        return "", new_line, figure_geo, figure_umap, {"label_selection": [], "umap_clickData": None}


# run app
if __name__ == '__main__':
    app.run_server(debug=True)
