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


def hide_all_traces():
    for trace in fig_geo.data:
        trace.visible = False

    for trace in fig_umap.data:
        trace.visible = False


# plot settings
scatter_size = 1.5

# dbscan labels
labels = pd.read_csv("../data/dbscan_labels.csv")

# heatmap data
data = pd.read_csv("../data/dbscan_scores_incomplete.csv")
groupby_cols = ['clustering_on', 'scores_on', 'eps', 'min_samples']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data[(data.clustering_on == 'embedding') & (data.scores_on == 'embedding')]  # filter
data = data.drop(['clustering_on', 'scores_on'], axis=1)
data = data.sort_values(["eps", "min_samples"])
data.nnoise = data.nnoise*100/49131
epss = np.sort(data.eps.unique())  # all epsilons
min_sampless = np.sort(data.min_samples.unique())  # all min_samples
all_combos = list(it.product(*[epss, min_sampless]))  # all combinations
score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski", "Davies-Bouldin": "davies_bouldin", "N clusters": "nclusters", "N noise": "nnoise"}
dx = abs(min_sampless[0] - min_sampless[1])
dy = abs(epss[0] - epss[1])

print("data loaded")

# current heatmap data
cur_score = "calinski"
field = pd.DataFrame(data[cur_score].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)

# figures
fig_geo = go.Figure()
fig_umap = go.Figure()
for eps, min_samples in all_combos:  # [[0.01, 3], [0.2, 3]]:  
    for n in ["noise", "no_noise"]:
        labels_filtered = labels[(labels.eps == eps) & (labels.min_samples == min_samples)]
        labels_filtered = color_code_labels(labels_filtered, label_name="label_embedding")

        if n == "no_noise":
            labels_filtered = labels_filtered[labels_filtered.label_embedding != -1]

        fig_geo.add_trace(
            go.Scatter3d(name=f"{eps}-{min_samples}-{n}-geo", x=labels_filtered.LONGITUDE, y=labels_filtered.LATITUDE, z=labels_filtered.LEV_M * -1,
                         mode='markers', marker=dict(size=scatter_size, color=labels_filtered.color, opacity=1), visible=False)
        )
        fig_umap.add_trace(
            go.Scatter3d(name=f"{eps}-{min_samples}-{n}-umap", x=labels_filtered.e0, y=labels_filtered.e1, z=labels_filtered.e2,
                         mode='markers', marker=dict(size=scatter_size, color=labels_filtered.color, opacity=1), visible=False)
        )
fig_geo.data[0].visible = True
fig_umap.data[0].visible = True

# heatmaps
heatmap_score = px.imshow(field, aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": cur_score},
                          color_continuous_scale='gray')
print("traces defined")

# current hyperparameter values
cur_eps = 0.01
cur_min_samples = 3

# dash app and layout
app = Dash(__name__)
app.layout = html.Div([
    html.Div([dcc.RadioItems(['N clusters', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette'], 'Calinski-Harabasz',
                             id='score', labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
              dcc.Checklist(["Hide noise"], [], id="hide-noise-check")
              ]),
    html.Div(dcc.Graph(figure=heatmap_score, id='heatmap-score',
                       clickData={'points': [{'x': min_sampless[0], 'y': epss[0], 'z': field.iloc[0, 0]}]}),
             style={'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_geo, id='fig-geo'),
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'}),
    html.Div(dcc.Textarea(id="textarea", value="Current parameters: "),
             style={'margin': dict(l=20, r=20, t=20, b=20), "width": "49%",
                    "height": "49%", "resize": "none", 'display': 'inline-block'}),
    html.Div(dcc.Graph(figure=fig_umap, id='fig-umap'),
             style={'margin': dict(l=20, r=20, t=20, b=20), "paper_bgcolor": "LightSteelBlue",
                    'display': 'inline-block'})
])


@app.callback(
    Output('textarea', 'value'),
    Output('heatmap-score', 'figure'),
    Output('fig-geo', 'figure'),
    Output('fig-umap', 'figure'),
    Input('score', 'value'),
    Input('textarea', 'value'),
    Input('heatmap-score', 'clickData'),
    Input('hide-noise-check', 'value'))
def update_heatmap(score, old_text, clickData, check_value):
    field = pd.DataFrame(data[score_map[score]].to_numpy().reshape(len(epss), len(min_sampless)),
                         index=epss, columns=min_sampless)

    if clickData:
        # get click coordinates
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        z = clickData['points'][0]['z']
        print(x, y, z)

        # Update the heatmap figure with a red frame around the selected point
        new_fig = px.imshow(field,  aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": score},
                            color_continuous_scale='gray')
        new_fig.update_layout(shapes=[
            go.layout.Shape(
                type='rect',
                x0=x-dx/2, y0=y-dy/2,
                x1=x+dx/2, y1=y+dy/2,
                line=dict(color='red', width=2)
            )
        ])

        # update label plots
        hide_all_traces()
        if not check_value:
            fig_geo.update_traces(visible=True, selector={'name': f"{y}-{x}-noise-geo"})
            fig_umap.update_traces(visible=True, selector={'name': f"{y}-{x}-noise-umap"})

        elif check_value == ["Hide noise"]:
            fig_geo.update_traces(visible=True, selector={'name': f"{y}-{x}-no_noise-geo"})
            fig_umap.update_traces(visible=True, selector={'name': f"{y}-{x}-no_noise-umap"})

        fig_geo.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")
        fig_umap.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="LightSteelBlue")

        return 'Current parameters: \neps = {}\nmin_samples = {}\n{} = {}'.format(y, x, score_map[score], z), new_fig, fig_geo, fig_umap
    else:
        return old_text, heatmap_score, fig_geo, fig_umap


# run app
if __name__ == '__main__':
    app.run_server(debug=True)
