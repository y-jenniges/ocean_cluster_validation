import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import pandas as pd
import numpy as np
import glasbey
from kneed import KneeLocator

# --- plotting utils -------------------------------------------------------------------- #
def plot_embedding(embedding, save_as=None):
    """ Plot a 3D embedding. """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.08, s=2, marker=".")
    plt.xlabel("Axis 0")
    plt.ylabel("Axis 1")
    ax.set_zlabel("Axis 2")
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
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(temp["e0"], temp["e1"], temp["e2"], c=temp["color"], alpha=0.8, zorder=4, s=1) 
        if save_dir:
            plt.savefig(save_dir + f"labels_in_umapspace{suffix}.png")
        plt.tight_layout()
        plt.show()
        

def color_code_labels(df, color_noise_black=True, drop_noise=True):
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
def plot_elbow_curve(df, knee=None, thresh=None, y_name="count", y_label="Log number of samples", save_dir=None, suffix=""):
    x = df.index
    y = df[y_name]

    fig = plt.figure()
    plt.plot(x, y, "-") 

    if thresh:
        closest = df.iloc[(df[y_name]-thresh).abs().argsort()[:1]]
        knee = closest.index[0]
        print(knee)
    elif knee:
        thresh = df.iloc[knee][y_name]
        
    plt.hlines(thresh, plt.xlim()[0], plt.xlim()[1], linestyles='dashed', color="orange")
    plt.vlines(knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color="orange")

    plt.xlabel('Clusters')
    plt.xticks([knee])
    plt.yticks(list(plt.yticks()[0]) + [thresh])
    plt.yscale("log")
    plt.ylabel(y_label)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + f"elbow_curve{suffix}.png")
        
    plt.show()

def compute_elbow_threshold(df_label_counts, y_name="count"):
    """ Compute knee/elbow of an L-curve using the Kneed library. Returns knee and its respective y-value. """
    x = df_label_counts.index
    y = df_label_counts[y_name]

    # compute knee
    kn = KneeLocator(x=x, y=y, curve='convex', direction='increasing')
    knee = kn.knee

    return knee, df_label_counts.iloc[knee][y_name]


def drop_clusters_with_few_samples(df, thresh=None, plotting=True, save_dir=None, suffix=""):
    """ If thresh is None, the Kneedle algorithm will be used to determine a treshold. """
    temp = df.copy()
    
    # count number of grid cells in each cluster
    df_nums = temp.groupby("label").count()["color"].reset_index().rename(columns={"color": "count"})
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

     # set all labels to -2  where num samples is too small
    labels_to_keep = list(df_nums[df_nums["count"] >= thresh].label)
    temp.loc[~(temp.label.astype(str).isin(labels_to_keep)), "label"] = -2

    # plotting
    coupled_label_plot(df=temp, color_label="color", save_dir=save_dir, suffix=suffix, umap_plot=True)
    
    return temp