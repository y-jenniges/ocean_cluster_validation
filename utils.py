import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import pandas as pd
import numpy as np
import glasbey
from kneed import KneeLocator
from geopy import distance  # geopy can only compute distances at surface
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance as scipy_distance
import gsw


# --- plotting utils -------------------------------------------------------------------- #
def plot_embedding(embedding, color_label=None, alpha=0.08, size=2, save_as=None):
    """ Plot a 2d or 3d embedding. (It can be a pandas.DataFrame or a numpy.array.) """
    # determine x, y, z, data and colour
    if isinstance(embedding, pd.DataFrame):
        x = embedding["e0"]
        y = embedding["e1"]
        z = embedding["e2"] if "e2" in embedding.columns else None
        c = embedding[color_label] if color_label is not None else None
    else:
        x = embedding[:, 0]
        y = embedding[:, 1]
        z = embedding[:, 2] if embedding.shape[1] == 3 else None
        c = color_label 

    # plot
    fig = plt.figure(figsize=(7, 6))

    # 3d
    if z is not None:
        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel("Axis 2")
        if c is None:
            ax.scatter(x, y, z, alpha=alpha, s=size, marker=".")
        else:
            ax.scatter(x, y, z, alpha=alpha, c=c, s=size, marker=".")
    # 2d
    else:
        if c is None:
            plt.scatter(x, y, alpha=alpha, s=size, marker=".")
        else:
            plt.scatter(x, y, alpha=alpha, c=c, s=size, marker=".")
             
    plt.xlabel("Axis 0")
    plt.ylabel("Axis 1")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def coupled_label_plot(df, color_label="color", save_dir=None, suffix="", umap_plot=True):
    """ Plots cluster labels in geographical and embedded space. """
    temp = df.copy()

    # define basemap
    mymap = Basemap(llcrnrlon=temp["LONGITUDE"].min(), llcrnrlat=temp["LATITUDE"].min(), 
                    urcrnrlon=temp["LONGITUDE"].max(), urcrnrlat=temp["LATITUDE"].max(), fix_aspect=False)

    # geographical plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(temp["LONGITUDE"], temp["LATITUDE"], temp["LEV_M"], c=temp[color_label], s=0.5, alpha=1, zorder=4) 
    ax.add_collection3d(mymap.drawcoastlines(linewidth=0.5))
    ax.set_box_aspect((np.ptp(temp["LONGITUDE"]), np.ptp(temp["LATITUDE"]), np.ptp(temp["LEV_M"])/50))  # aspect ratio is 1:1:1 in data space
    plt.gca().invert_zaxis()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + f"labels_in_geospace{suffix}.png")
    plt.show()

    # UMAP plot
    if umap_plot:
        save_as = save_dir + f"labels_in_umapspace{suffix}.png" if save_dir is not None else None
        plot_embedding(temp, color_label=color_label, alpha=1, save_as=save_as)


def color_code_labels(df, color_noise_black=False, drop_noise=False, column_name="label"):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # define colors
    unique_labels = np.sort(np.unique(temp[column_name]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp[column_name].map(lambda x: color_map[x])

    # how to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp[column_name] == -1, "color"] = "#000000"
    if drop_noise:
        temp = temp[temp[column_name] != -1]
        
    return temp


def plot_ts(df, figsize=(4, 4), xlim=None, save_as=None):
    """ Plot T-S diagram. """
    temp = df.copy()
    
    # compute necessary parameters
    temp["pressure"] = gsw.p_from_z(-1*temp["LEV_M"], temp["LATITUDE"])
    temp["abs_salinity"] = gsw.SA_from_SP(temp["P_SALINITY"], temp["pressure"], temp["LONGITUDE"], temp["LATITUDE"])
    temp["cons_temperature"] = gsw.CT_from_pt(temp["abs_salinity"], temp["P_TEMPERATURE"])
    temp["rho"] = gsw.rho(temp["abs_salinity"], temp["cons_temperature"], temp["pressure"])

    # limits
    smin = temp["abs_salinity"].min() - (0.01 * temp["abs_salinity"].min())
    smax = temp["abs_salinity"].max() + (0.01 * temp["abs_salinity"].max())
    tmin = temp["cons_temperature"].min() - (0.1 * temp["cons_temperature"].max())
    tmax = temp["cons_temperature"].max() + (0.1 * temp["cons_temperature"].max())
    
    # number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) + 1, 0))
    
    # empty grid
    dens = np.zeros((ydim, xdim))
    
    # temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) + tmin
    
    # fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)
    
    # convert to sigma-t
    dens = dens - 1000
    
    # basemap for plot
    map = Basemap(llcrnrlon=temp["LONGITUDE"].min(), llcrnrlat=temp["LATITUDE"].min(),
                  urcrnrlon=temp["LONGITUDE"].max(), urcrnrlat=temp["LATITUDE"].max())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    CS = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(CS, fontsize=12, inline=1, fmt='%1.0f')  # Label every second level
    ax.scatter(x=temp["abs_salinity"], y=temp["cons_temperature"], s=9, c=temp["color"], alpha=1, marker=".")
    ax.set_xlabel('Absolute salinity [k/kg]')
    ax.set_ylabel('Conservative temperature [°C]')
    ax.set_xlim(xlim)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()
    

# --- utils to remove clusters by grid cell number -------------------------------------------------------------------- #
def plot_elbow_curve(df, knee=None, thresh=None, y_name="count", y_label="Log number of grid cells", y_scale="log", save_dir=None, suffix=""):
    x = df.index
    y = df[y_name]

    fig = plt.figure()
    plt.plot(x, y, "-") 

    if thresh:
        closest = df.iloc[(df[y_name]-thresh).abs().argsort()[:1]]
        knee = closest.index[0]
    elif knee:
        thresh = df.iloc[knee][y_name]
        
    plt.hlines(thresh, plt.xlim()[0], plt.xlim()[1], linestyles='dashed', color="orange")
    plt.vlines(knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color="orange")

    plt.xlabel('Cluster')
    plt.xticks([knee])
    plt.text(0, thresh + thresh/100*5, thresh, color="orange")
    plt.yscale(y_scale)
    plt.ylabel(y_label)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + f"elbow_curve{suffix}.png")
        
    plt.show()


def compute_elbow_threshold(df_label_counts, y_name="count", slope_direction="increasing"):
    """ Compute knee/elbow of an L-curve using the Kneed library. Returns knee and its respective y-value. """
    x = df_label_counts.index
    y = df_label_counts[y_name]

    # compute knee
    kn = KneeLocator(x=x, y=y, curve='convex', direction=slope_direction)
    knee = kn.knee

    return knee, df_label_counts.iloc[knee][y_name]


def drop_clusters_with_few_samples(df, thresh=None, plotting=True, y_label="Log number of grid cells", y_scale="log", save_dir=None, suffix=""):
    """ If thresh is None, the Kneedle algorithm will be used to determine a treshold. """
    temp = df.copy()
    knee = None
    
    # count number of grid cells in each cluster
    some_column = df.columns[0]
    df_nums = temp.groupby("label").count()[some_column].reset_index().rename(columns={some_column: "count"})
    df_nums = df_nums.sort_values("count").reset_index(drop=True)
    df_nums["label"] = df_nums["label"].astype(str)

    # compute threshold to cut off clusters
    if not thresh:
        knee, thresh = compute_elbow_threshold(df_nums, y_name="count")
        if plotting:
            plot_elbow_curve(df_nums, knee=knee, y_label=y_label, y_scale=y_scale, save_dir=save_dir, suffix=suffix)
    elif plotting:
        # plot number of cells per cluster
        plot_elbow_curve(df_nums, thresh=thresh, y_label=y_label, y_scale=y_scale, save_dir=save_dir, suffix=suffix)

     # set all labels to -1  where num samples is too small
    labels_to_keep = list(df_nums[df_nums["count"] >= thresh].label)
    temp.loc[~(temp.label.astype(str).isin(labels_to_keep)), "label"] = -1
    print("Remaining number of clusters: " + str(len(labels_to_keep)))
    
    return temp, knee, thresh, df_nums


# --- utils to remove clusters by grid cell volume -------------------------------------------------------------------- #
def compute_volume(row, dlat, dlon, depths):
    """ Computes the volume of a given grid cell. """
    # careful!! I am assuming that depth does not affect distance calculations
    # approximate volume as a box, ignore depth, surface approximated by trapezoid
    lat = row["LATITUDE"]
    lon = row["LONGITUDE"]
    depth = row["LEV_M"]

    # find depth step size
    idx = np.argwhere(depths == depth)
    next_depth = depths[idx+1]
    ddepth = (next_depth - depth).flatten()[0]

    # compute box edge points
    p0 = np.array([lat - dlat / 2, lon - dlon / 2, depth])
    p1 = np.array([lat - dlat / 2, lon + dlon / 2, depth])
    p2 = np.array([lat + dlat / 2, lon + dlon / 2, depth])
    p3 = np.array([lat + dlat / 2, lon - dlon / 2, depth])

    # compute distances
    # d03 = distance.distance(p0, p3).m
    d01 = distance.distance(p0, p1).m
    d23 = distance.distance(p2, p3).m
    # d12 = distance.distance(p2, p1).m

    # compute height of trapezoid
    p01 = np.array([p0[0], lon, p0[2]])
    p23 = np.array([p2[0], lon, p2[2]])
    h = distance.distance(p01, p23).m

    # compute volume
    # v = round((depth+ ddepth) * 0.5 * (d01 + d23) * h, 4)  # volume in m3
    v = ddepth * 0.5 * (d01 + d23) * h  # volume in m3

    return v


def drop_clusters_with_small_volume(df, thresh=None, y_label="Log number of grid cells", y_scale="log", save_dir=None, suffix="", plotting=True):
    """ If thresh is None, the Kneedle algorithm will be used to determine a treshold. """
    temp = df.copy()
    knee = None

    # define resolution
    dlat = 1
    dlon = 1
    
    # get depth resolution
    depths = temp["LEV_M"].unique()
    depths = np.concatenate([depths, np.array([5000]).reshape(1, -1).flatten()])  # adding the lower depth bound

    # compute volume
    temp["volume"] = temp.apply(compute_volume, axis=1, args=(dlat, dlon, depths))  # careful with rounding

    # check cluster volumes
    df_vols = temp.groupby("label").sum()["volume"].reset_index()
    df_vols = df_vols.sort_values("volume").reset_index(drop=True)
    df_vols["label"] = df_vols["label"].astype(str)

    # compute threshold to cut off clusters
    if not thresh:
        knee, thresh = compute_elbow_threshold(df_vols, y_name="volume")
        if plotting:
            plot_elbow_curve(df_vols, knee=knee, y_name="volume", y_label=y_label, y_scale=y_scale, save_dir=save_dir, suffix=suffix)
    elif plotting:
        # plot volume per cluster
        plot_elbow_curve(df_vols, thresh=thresh, y_name="volume", y_label=y_label, y_scale=y_scale, save_dir=save_dir, suffix=suffix)

     # set all labels to -1  where num samples is too small
    labels_to_keep = list(df_vols[df_vols["volume"] >= thresh].label)
    print("Remaining number of clusters: " + str(len(labels_to_keep)))
    temp.loc[~(temp.label.astype(str).isin(labels_to_keep)), "label"] = -1

    return temp, knee, thresh, df_vols


# --- utils to remove clusters by geographic cohesion ------------------------------------------------------------------- #
def find_cell_neighbours(table, cell_idx, depths, dlat=1, dlon=1, known_neighbours=[]):
    """Recursively find geographically neighbouring cells of a given cell. """
    # init neighbourhood
    hood = [cell_idx]

    # reference point
    cell_lat = round(table.loc[cell_idx].LATITUDE, 1)
    cell_lon = round(table.loc[cell_idx].LONGITUDE, 1)
    cell_depth = round(table.loc[cell_idx].LEV_M)
    depth_idx = np.argwhere(depths == cell_depth)[0][0]

    # find geographic neighbours of reference point
    for latitude in [cell_lat, cell_lat + dlat, cell_lat - dlat]:
        for longitude in [cell_lon, cell_lon + dlon, cell_lon - dlon]:
            for depth_index in [depth_idx, depth_idx + 1, depth_idx - 1]:
                # do not compare to self
                if not (latitude == cell_lat and longitude == cell_lon and depth_index == depth_idx):
                    if depth_index in range(0, len(depths)):
                        d = depths[depth_index]
                        new_neighbour = table[(np.round(table.LATITUDE, 1) == latitude) &
                                              (np.round(table.LONGITUDE, 1) == longitude) &
                                              (np.round(table.LEV_M) == d)]
                        if not new_neighbour.empty:
                            if new_neighbour.index not in known_neighbours:
                                hood.append(new_neighbour.index[0])

    # recursively find neighbours of the neighbours
    for new_idx in hood:
        # do not visit self again
        if new_idx != cell_idx:
            hood = list(set(hood + find_cell_neighbours(table=table, cell_idx=new_idx, depths=depths, dlat=dlat, dlon=dlon, known_neighbours=hood + known_neighbours)))

    return hood


def find_neighbours(df, dlat=1, dlon=1):
    """ Fore each cluster, find all grid cells that are geographically cohesive. Each sub-cluster forms a neighbourhood. 
        Example usage: 
        df = pd.DataFrame({"LATITUDE": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                           "LONGITUDE": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                           "LEV_M": 16*[1],
                           "label": [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1]})
        neighbours = find_neighbours(df, dlat=1, dlon=1)
    """
    # determine resolution
    depths = df.LEV_M.unique()
    
    # iterate over all clusters in the clustering
    res = []
    for c in [x for x in df.label.unique() if x != -1]:
        print("Computing neighbours of cluster ", c)

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
                neighbourhood = np.sort(find_cell_neighbours(table=temp, cell_idx=i, depths=depths, dlat=dlat, dlon=dlon, known_neighbours=[]))
                neighbourhoods.append(neighbourhood)
                visited_points = list(set(visited_points + list(neighbourhood)))

                # determine neighbourhood weight (as standard deviation, i.e. deviation from mean position)
                local_mean = temp_scaled.loc[neighbourhood][["LATITUDE", "LONGITUDE", "LEV_M"]].mean()
                distance_weight = scipy_distance.euclidean(cluster_mean, local_mean)

                # print(f"    visited point: {i}, neighbourhood: {neighbourhood}, distance_weight: {distance_weight}")

                res.append(pd.DataFrame({"visited_point": [i],
                                         "neighbourhood": [neighbourhood],
                                         "distance_weight": [distance_weight],
                                         "cluster": [c]}))

                score = score + distance_weight
    res = pd.concat(res)

    return res
    