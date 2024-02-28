import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import pandas as pd
import numpy as np
import glasbey
from kneed import KneeLocator
from geopy import distance  # geopy can only compute distances at surface


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


def color_code_labels(df, color_noise_black=False, drop_noise=False):
    """ Add a color for each label in the clustering using the Glasbey library. """
    temp = df.copy()

    # define colors
    unique_labels = np.sort(np.unique(temp["label"]))
    colors = glasbey.create_palette(palette_size=len(unique_labels))
    color_map = {label: color for label, color in zip(unique_labels, colors)}
    temp["color"] = temp["label"].map(lambda x: color_map[x])

    # how to deal with -1 labels (which is noise in DBSCAN)
    if color_noise_black:
        temp.loc[temp["label"] == -1, "color"] = "#000000" 
    if drop_noise:
        temp = temp[temp["label"] != -1]
        
    return temp


# --- utils to remove clusters -------------------------------------------------------------------- #
def plot_elbow_curve(df, knee=None, thresh=None, y_name="count", y_label="Log number of grid cells", save_dir=None, suffix="", y_scale="log"):
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

    plt.xlabel('Clusters')
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


def drop_clusters_with_few_samples(df, thresh=None, plotting=True, save_dir=None, suffix=""):
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
            plot_elbow_curve(df_nums, knee=knee, save_dir=save_dir, suffix=suffix)
    elif plotting:
        # plot number of cells per cluster
        plot_elbow_curve(df_nums, thresh=thresh, save_dir=save_dir, suffix=suffix)

     # set all labels to -1  where num samples is too small
    labels_to_keep = list(df_nums[df_nums["count"] >= thresh].label)
    temp.loc[~(temp.label.astype(str).isin(labels_to_keep)), "label"] = -1
    print("Remaining number of clusters: " + str(len(labels_to_keep)))
    
    return temp, knee, thresh, df_nums

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
    
def drop_clusters_with_small_volume(df, thresh=None, y_scale="linear", save_dir=None, suffix="", plotting=True):
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
            plot_elbow_curve(df_vols, knee=knee, y_name="volume", y_label="Log cluster volume", save_dir=save_dir, suffix=suffix)
    elif plotting:
        # plot volume per cluster
        plot_elbow_curve(df_vols, thresh=thresh, y_name="volume", y_label="Log cluster volume", save_dir=save_dir, suffix=suffix)

    # decide which labels to keep
    labels_to_keep = df_vols[df_vols["volume"] >= thresh].label
    print("Remaining number of clusters: " + str(len(labels_to_keep)))
    df_new = temp[temp["label"].astype(str).isin(labels_to_keep)]

    # plot new data in 3D and UMAP space
    coupled_label_plot(df_new[df_new.label != -1])

    return df_new, knee, thresh, df_vols
