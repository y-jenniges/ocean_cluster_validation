import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd

# dbscan labels
# labels = pd.read_csv("data/dbscan_labels.csv")

# heatmap data
data = pd.read_csv("data/dbscan_scores_incomplete.csv")
groupby_cols = ['clustering_on', 'scores_on', 'eps', 'min_samples']
data = data.groupby(groupby_cols).mean().drop("iteration", axis=1).reset_index()  # average over iterations
data = data[(data.clustering_on == 'embedding') & (data.scores_on == 'embedding')]  # filter
data = data.drop(['clustering_on', 'scores_on'], axis=1)
data = data.sort_values(["eps", "min_samples"])
data.nnoise = data.nnoise*100/49131

epss = np.sort(data.eps.unique())
min_sampless = np.sort(data.min_samples.unique())
score = "calinski"
score_map = {"Silhouette": "silhouette", "Calinski-Harabasz": "calinski", "Davies-Bouldin": "davies_bouldin", "N clusters": "nclusters", "N noise": "nnoise"}
dx = abs(min_sampless[0] - min_sampless[1])
dy = abs(epss[0] - epss[1])
field = pd.DataFrame(data[score].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)
field_noise = pd.DataFrame(data["nnoise"].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)
# labels_filtered = labels[(labels.eps == epss[0]) & (labels.min_samples == min_sampless[0])]

# Create Dash app
app = Dash(__name__)

# create heatmap figures
fig = px.imshow(field, aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": score}, color_continuous_scale='gray')
fig_noise = px.imshow(field_noise, aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": "N noise [%]"}, color_continuous_scale='gray')

# app layout
app.layout = html.Div([
    dcc.RadioItems(['N clusters', 'Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette'], 'Calinski-Harabasz', id='score', labelStyle={'display': 'inline-block', 'marginTop': '5px'}),
    dcc.Graph(figure=fig, id='heatmap', hoverData={'points': [{'x': min_sampless[0], 'y': epss[0], 'z': field.iloc[0, 0]}]}),
    dcc.Graph(figure=fig_noise, id='heatmap_noise'),
    html.Div(id='selected-value'),
])


# Callback to handle selection and update the red frame
@app.callback(
    Output('selected-value', 'children'),
    Output('heatmap', 'figure'),
    Output('heatmap_noise', 'figure'),
    Input('score', 'value'),
    Input('heatmap', 'hoverData'))
def update_heatmap(score, hoverData):
    field = pd.DataFrame(data[score_map[score]].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)

    if hoverData:
        # get hover coordinates
        x = hoverData['points'][0]['x']
        y = hoverData['points'][0]['y']
        z = hoverData['points'][0]['z']
        noise = np.round(field_noise.loc[y, x], 1)

        # Update the heatmap figure with a red frame around the selected point
        new_fig = px.imshow(field,  aspect="auto", width=1000, labels={"x": "min_samples", "y": "eps", "color": score}, color_continuous_scale='gray')
        new_fig.update_layout(shapes=[
            go.layout.Shape(
                type='rect',
                x0=x-dx/2, y0=y-dy/2,
                x1=x+dx/2, y1=y+dy/2,
                line=dict(color='red', width=2)
            )
        ])

        new_fig_noise = fig_noise
        fig_noise.update_layout(shapes=[
            go.layout.Shape(
                type='rect',
                x0=x-dx/2, y0=y-dy/2,
                x1=x+dx/2, y1=y+dy/2,
                line=dict(color='red', width=2)
            )
        ])

        return f'min_samples = {x}</br>eps = {np.round(y, 8)}\nnnoise = {noise}%\n{score} = {np.round(z, 2)}', new_fig, new_fig_noise

    else:
        return 'Hover over the heatmap to select a point.', fig, fig_noise


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
