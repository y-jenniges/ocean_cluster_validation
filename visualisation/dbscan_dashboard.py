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
labels = pd.read_csv("../data/dbscan_labels.csv")

# heatmap data
data = pd.read_csv("../data/dbscan_scores_incomplete.csv")
groupby_cols = ['clustering_on', 'scores_on', 'eps', 'min_samples']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data[(data.clustering_on == 'embedding') & (data.scores_on == 'embedding')]  # filter
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

# current heatmap data
cur_score = "calinski"
field = pd.DataFrame(data[cur_score].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)

# figures and heatmaps
fig_geo = go.Figure()
fig_umap = go.Figure()
fig_depth = go.Figure()
heatmap_score = px.imshow(field, aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": cur_score},
                          color_continuous_scale='gray')
# current hyperparameter values
cur_eps = 0.01
cur_min_samples = 2

# # create 2d scatter traces for each depth step
# for step in labels.LEV_M.unique():
#     temp = labels[(labels.eps == cur_eps) & (labels.min_samples == cur_min_samples) & (labels.LEV_M == step)]
#     temp = color_code_labels(temp, label_name=data_label)
#     fig_depth.add_trace(go.Scatter(visible=False, x=temp.LONGITUDE, y=temp.LATITUDE, name=f"{step}-depth",
#                                    mode='markers', marker=dict(size=scatter_size, color=temp.color, opacity=1)))
# fig_depth.data[0].visible = True


print("figures defined")

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div([dcc.RadioItems(['N clusters', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'N noise'],
                             'Calinski-Harabasz',
                             id='score', labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
              dcc.Checklist(["Hide noise"], ["Hide noise"], id="hide-noise-check")
              ]),
    html.Div(dcc.Graph(figure=heatmap_score, id='heatmap-score',
                       clickData={'points': [{'x': min_sampless[0], 'y': epss[0], 'z': field.iloc[0, 0]}]}),
             style={'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_geo, id='fig-geo'),
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),
    html.Div([dcc.Textarea(id="textarea", value="Current parameters: "),
              dcc.Graph(figure=fig_depth, id="fig-depth"),
              dcc.Slider(0, len(labels.LEV_M.unique())-1, value=0, id='depth-slider')],
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%",
                    "height": "49%", "resize": "none", 'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap'),
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),

    dcc.Store(id="cur-params", data={"eps": cur_eps, "min_samples": cur_min_samples})
])


# @app.callback(
#     Output('fig-depth', 'figure'),
#     Input('depth-slider', 'value'),
# )
# def update_2d_scatterplot(depth):
#     print("callback 2d scatter")
#     # udpate 2d scatter plot according to slider value
#     for trace in fig_depth.data:
#         trace.visible = False
#     fig_depth.update_traces(visible=True, selector={'name': f"{depth}-depth"})
#
#     return fig_depth


@app.callback(
    Output('textarea', 'value'),
    Output('heatmap-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Output('cur-params', 'data'),

    Input('score', 'value'),
    Input('heatmap-score', 'clickData'),
    Input('hide-noise-check', 'value'),
    Input('cur-params', 'data'),
    Input('fig-geo', 'figure'),
    Input('fig-umap', 'figure'),
)
def update_heatmap(score, clickData, check_value, cur_params, figure_geo, figure_umap):
    # update heatmap
    new_field = pd.DataFrame(data[score_map[score]].to_numpy().reshape(len(epss), len(min_sampless)),
                             index=epss, columns=min_sampless)
    eps = cur_params["eps"]
    min_samples = cur_params["min_samples"]
    score_value = new_field.loc[eps, min_samples]

    if clickData:
        print(clickData)
        # get click coordinates
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        score_value = new_field.loc[y, x]

        # Update the heatmap figure with a red frame around the selected point
        new_fig = px.imshow(new_field, aspect="auto", width=1000,
                            labels={"x": "min_samples", "y": "eps", "color": score},
                            color_continuous_scale='gray')
        new_fig.update_layout(shapes=[
            go.layout.Shape(
                type='rect',
                x0=x - dx / 2, y0=y - dy / 2,
                x1=x + dx / 2, y1=y + dy / 2,
                line=dict(color='red', width=2)
            )
        ])

        # update label plots
        cur_labels = labels[(labels.eps == y) & (labels.min_samples == x)]
        cur_labels = color_code_labels(cur_labels, label_name=data_label)

        if not check_value:
            n = "noise"
        elif check_value == ["Hide noise"]:
            n = "no_noise"
            cur_labels = cur_labels[cur_labels[data_label] != -1]

        figure_geo = go.Figure(data=go.Scatter3d(name=f"{y}-{x}-{n}-geo",
                                                 x=cur_labels.LONGITUDE, y=cur_labels.LATITUDE, z=cur_labels.LEV_M * -1,
                                                 mode='markers',
                                                 marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_umap = go.Figure(data=go.Scatter3d(name=f"{y}-{x}-{n}-umap",
                                                  x=cur_labels.e0, y=cur_labels.e1, z=cur_labels.e2,
                                                  mode='markers',
                                                  marker=dict(size=scatter_size, color=cur_labels.color, opacity=1)))
        figure_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")
        figure_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")

        return 'Current parameters: \neps = {}\nmin_samples = {}\n{} = {}'.format(
            y, x, score_map[score], np.round(score_value, 2)), \
               new_fig, figure_geo, figure_umap, {"eps": y, "min_samples": x}
    else:
        return 'Current parameters: \neps = {}\nmin_samples = {}\n{} = {}'.format(
            eps, min_samples, score_map[score], np.round(score_value, 2)), \
               heatmap_score, figure_geo, figure_umap, {"eps": eps, "min_samples": min_samples}


# run app
if __name__ == '__main__':
    app.run_server(debug=True)
