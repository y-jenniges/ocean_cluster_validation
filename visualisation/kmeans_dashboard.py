# no traces but data loading

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
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
data = data[(data.clustering_on == 'embedding') & (data.scores_on == 'embedding')]  # filter
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
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),
    html.Div([dcc.Textarea(id="textarea", value="Current parameters: "),
              # dcc.Graph(figure=fig_depth, id="fig-depth"),
              # dcc.Slider(0, len(labels.LEV_M.unique())-1, value=0, id='depth-slider')
              ],
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%",
                    "height": "49%", "resize": "none", 'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap'),
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),

    # dcc.Store(id="cur-params", data={"n_clusters": cur_n_clusters})
])


@app.callback(
    Output('textarea', 'value'),
    Output('line-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),

    Input('score', 'value'),
    Input('line-score', 'clickData'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
)
def update_heatmap(score, clickData, figure_geo, figure_umap):
    # update heatmap
    new_line = px.line(data, x='n_clusters', y=score_map[score], markers=True)
    new_line.update_traces(marker=dict(color=['red'] * len(n_clusterss)))

    if clickData:
        print(clickData)
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

        figure_geo = go.Figure(data=go.Scatter3d(name=f"{x}-geo",
                                                 x=cur_labels.LONGITUDE, y=cur_labels.LATITUDE, z=cur_labels.LEV_M * -1,
                                                 mode='markers',
                                                 marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_umap = go.Figure(data=go.Scatter3d(name=f"{x}-umap",
                                                  x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                                  mode='markers',
                                                  marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")
        figure_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")

        return 'Current parameters: \nn_clusters = {}\n{} = {}'.format(x, score_map[score], np.round(score_value, 2)), new_line, figure_geo, figure_umap
    else:
        return "", new_line, figure_geo, figure_umap


# run app
if __name__ == '__main__':
    app.run_server(debug=True)
