import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
import matplotlib.pyplot as plt
import seaborn as sns

# measure how geographically cohesive a cluster is

df = pd.read_csv("C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/uncertainty/umap_dbscan_7.csv")
df_drop = pd.read_csv(
    "C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/uncertainty/umap_dbscan_manualkneedrop_7.csv")

# resolution
dlat = 1
dlon = 1
depths = df.LEV_M.unique()


def find_neighbours(table, idx, known_neighbours=[]):
    # init neighbourhood
    hood = [idx]

    # reference point
    la = round(table.loc[idx].LATITUDE, 1)
    lo = round(table.loc[idx].LONGITUDE, 1)
    de = round(table.loc[idx].LEV_M)
    de_idx = np.argwhere(depths == de)[0][0]

    # find geographic neighbours of reference point
    for latitudee in [la, la + dlat, la - dlat]:
        for longitudee in [lo, lo + dlon, lo - dlon]:
            for depth_indexx in [de_idx, de_idx + 1, de_idx - 1]:
                # do not compare to self
                if not (latitudee == la and longitudee == lo and depth_indexx == de_idx):
                    # print("    not self")
                    if depth_indexx in range(0, len(depths)):
                        # print(latitudee, longitudee, depth_indexx)
                        # print("    depth found in index list")
                        d = depths[depth_indexx]
                        new_neighbour = table[(np.round(table.LATITUDE, 1) == latitudee) &
                                              (np.round(table.LONGITUDE, 1) == longitudee) &
                                              (np.round(table.LEV_M) == d)]
                        if not new_neighbour.empty:
                            # print("    not empty")
                            # print(neighbour)
                            if new_neighbour.index not in known_neighbours:
                                hood.append(new_neighbour.index[0])

    # recursively find neighbours of the neighbours
    # ns = [hood]
    for new_idx in hood:
        # do not visit self again
        if new_idx != idx:
            hood = list(set(hood + find_neighbours(table=table, idx=new_idx, known_neighbours=hood + known_neighbours)))
            # ns.append(hood)

    return hood


# # example
# df = pd.DataFrame({"LATITUDE": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
#                    "LONGITUDE": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#                    "LEV_M": 16*[1],
#                    "label": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]})
# temp = df[df.label == 1]
# depths = df.LEV_M.unique()
# dlat = 1
# dlon = 1

# iterate over all clusters in the clustering
res = []
for c in [x for x in df.label.unique() if x != -1]:
    print("Cluster ", c)

    # load cluster
    temp = df[df.label == c].sort_values(["LATITUDE", "LONGITUDE", "LEV_M"])
    all_idxs = list(temp.index)
    num_grid_cells = len(temp)

    # scale dataframe
    temp_scaled = pd.DataFrame(MinMaxScaler().fit_transform(temp), columns=temp.columns, index=temp.index)
    cluster_mean = temp_scaled[["LATITUDE", "LONGITUDE", "LEV_M"]].mean()

    # iterate over all points of current cluster
    visited_points = []
    score = 0
    neighbourhoods = []
    for i in list(temp.index):
        # make sure you did not visit this point before
        if i not in visited_points:
            neighbourhood = np.sort(find_neighbours(table=temp, idx=i, known_neighbours=[]))
            neighbourhoods.append(neighbourhood)
            visited_points = list(set(visited_points + list(neighbourhood)))

            # determine neighbourhood weight (as standard deviation, i.e. deviation from mean position)
            local_mean = temp_scaled.loc[neighbourhood][["LATITUDE", "LONGITUDE", "LEV_M"]].mean()
            distance_weight = distance.euclidean(cluster_mean, local_mean)

            print(f"    visited point: {i}, neighbourhood: {neighbourhood}, distance_weight: {distance_weight}")

            res.append(pd.DataFrame({"visited_point": [i],
                                     "neighbourhood": [neighbourhood],
                                     "distance_weight": [distance_weight],
                                     "cluster": [c]}))

            score = score + distance_weight
    print(f"    {score}")
    print()

    # res.append(pd.DataFrame({"neighbourhoods": [neighbourhoods],
    #                          "score": [score],
    #                          "num_sections": [len(neighbourhoods)],
    #                          "cluster": [c]}))

# score reflects geographic scatter and in how many sections the cluster splits
res = pd.concat(res)
res.to_csv("cluster_analysis.csv", index=False)

# res = pd.DataFrame({"score": [0.5, 1, 3, 1, 4, 14], "num_sections": [1, 1, 1, 2, 3, 2], "cluster": [0, 1, 2, 3, 4, 5]})
res.cluster = res["cluster"].astype(str)
res = res.sort_values("num_sections")

sns.scatterplot(res, x="cluster", y="num_sections", hue="score")
plt.savefig("cluster_analysis.png")
plt.show(block=True)


######
import pandas as pd
df = pd.read_csv("C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/uncertainty/umap_dbscan_7.csv")
analysis = pd.read_csv("cluster_analysis.csv")

# case A: drop small clusters
# case B: drop small-volume clusters
# case C: Drop incohesive clusters
# case D: Drop 1-cell neighbourhoods


##################################################################################################################
# --- re-assign unlabeled grid cells
import pandas as pd
import numpy as np

# df = pd.read_csv("C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/uncertainty/umap_dbscan_7.csv")
df_drop = pd.read_csv(
    "C:/Users/yvjennig/Downloads/output_final/output_final/dbscan/uncertainty/umap_dbscan_manualkneedrop_7.csv")
temp = df_drop[df_drop.label == -1]

depths = df_drop.LEV_M.unique()
dlat = 1
dlon = 1

new_df = []
for row in temp.iterrows():
    print(row[0])

    # reference point
    lat = round(row[1].LATITUDE, 1)
    lon = round(row[1].LONGITUDE, 1)
    dep = round(row[1].LEV_M)
    dep_idx = np.argwhere(depths == dep)[0][0]

    # search the closest data points until there are neighbours with a label
    neighbour_found = False
    search_radius = 1
    while not neighbour_found:
        # find geographically closest points (within search radius)
        neighbours = []
        lat_step = dlat * search_radius
        lon_step = dlon * search_radius
        dep_step = dep_idx * search_radius

        for latitude in [lat, lat + lat_step, lat - lat_step]:
            for longitude in [lon, lon + lon_step, lon - lon_step]:
                for depth_index in [dep_idx, dep_idx + dep_step, dep_idx - dep_step]:
                    # do not compare to self
                    # print(latitude, longitude, depth_index)
                    if not (latitude == lat and longitude == lon and depth_index == dep_idx):
                        # print("    not self")
                        if depth_index in range(0, len(depths)):
                            depth = depths[depth_index]
                            neighbour = df_drop[(round(df_drop.LATITUDE, 1) == latitude) &
                                                (round(df_drop.LONGITUDE, 1) == longitude) &
                                                (round(df_drop.LEV_M) == depth)]
                            if not neighbour.empty:
                                if neighbour.iloc[0].label != -1:  # use only neighbours that have valid labels
                                    neighbours.append(neighbour)
        if neighbours:
            neighbours = pd.concat(neighbours)
            neighbour_found = True

            # determine labels of neighbours
            neighbour_labels = list(neighbours.label)

            # determine most frequent label
            max_label = max(neighbour_labels, key=neighbour_labels.count)

            # use most frequent label for label of current data point
            df_drop.loc[row[0], "label"] = max_label
            new_df.append(pd.DataFrame({"idx": [row[0]], "new_label": [max_label]}))

        # increase search radius
        search_radius = search_radius + 1

new_df = pd.concat(new_df)

df_drop.to_csv("re_assigned.csv", index=False)
