import plotly.graph_objects as go
import pandas as pd
import numpy as np
import itertools as it
# code: https://plotly.com/python/custom-buttons/

# load dataset
# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")
data_label = "label_embedding"
# data_label = "label_original"

# heatmap data
data = pd.read_csv("../output_final/dbscan_scores.csv")
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

# current heatmap data
cur_score = "davies_bouldin"
field = pd.DataFrame(data[cur_score].to_numpy().reshape(len(epss), len(min_sampless)), index=epss, columns=min_sampless)

# create figure
fig = go.Figure()

# Add surface trace
fig.add_trace(go.Surface(z=field, colorscale="gray"))

# Update plot sizing
fig.update_layout(
    width=800,
    height=600,
    autosize=False,
    margin=dict(t=0, b=0, l=0, r=0),
    template="plotly_white",
)

# Update 3D scene options
fig.update_scenes(
    aspectratio=dict(x=1, y=1, z=0.7),
    aspectmode="manual"
)

# Add dropdown
fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=["type", "surface"],
                    label="3D Surface",
                    method="restyle"
                ),
                dict(
                    args=["type", "heatmap"],
                    label="Heatmap",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.11,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

# Add annotation
fig.update_layout(
    annotations=[
        dict(text="Trace type:", showarrow=False, x=0, y=1.08, yref="paper", align="left")
    ])

fig.show()
