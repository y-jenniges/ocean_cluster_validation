import os.path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import gsw
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import utils
import time
# import plotly.io as pio
# pio.renderers.default = "browser"
# import plotly.graph_objs as go
# import plotly.express as px
import matplotlib
from matplotlib.colors import hex2color
from geopy import distance
import math

import dask
from dask import delayed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def compute_area(row, dlat, dlon):
    """ Compute surface area of a grid cell. """
    lat = row["LATITUDE"]
    lon = row["LONGITUDE"]

    R = 6378137.0  # earth radius in meters (WGS84 standard)

    lat_length = 111320  # average meters per degree of latitude
    lon_length = (math.pi / 180) * R * math.cos(
        math.radians(lat))  # length of 1 degree of longitude in meters at given latitude

    area = dlat * lat_length * dlon * lon_length
    return area


# ~2000 times faster than original compute_volume, very similar outcome (std/mean of the difference = 1.4)
def compute_volume_alternative(row, dlat, dlon, depths):
    """ Computes the volume of a grid cell. """
    lat = row["LATITUDE"]
    lon = row["LONGITUDE"]
    depth = row["LEV_M"]

    # find depth step size
    idx = np.argwhere(depths == depth)
    next_depth = depths[idx + 1]
    ddepth = (next_depth - depth).flatten()[0]

    R = 6378137.0  # earth radius in meters (WGS84 standard)

    lat_length = 111320  # average meters per degree of latitude
    lon_length = (math.pi / 180) * R * math.cos(
        math.radians(lat))  # length of 1 degree of longitude in meters at given latitude

    volume = dlat * lat_length * dlon * lon_length * ddepth
    return volume


def nemi_function_volume(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """
    base_id = base_id

    # list of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(nemi_pack))]
    compare_ids.pop(base_id)

    # identify clusters from the base ensemble member
    base_labels = [x for i, x in nemi_pack if int(i) == int(base_id)][0].sorted_label  # .label.unique())
    base_volumes = [x for i, x in nemi_pack if int(i) == int(base_id)][0].volume

    # number of clusters
    num_clusters = int(np.max(base_labels) + 1)

    # if not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0])) * np.nan

    # print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [(nemi[1].sorted_label, nemi[1].volume) for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels, compare_volumes = dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1 == base_labels)] = 1
            # # Count numer of entries [Why?]
            summaryStats[0, c1] = np.sum(data1_M)
            # mark volumes of the base cluster
            data1_volumes = data1_M * base_volumes

            # go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2 == compare_labels)] = 1

                # mark volumes of the comparison cluster
                data2_volumes = data2_M * compare_volumes

                # Sum of flags where the two datasets of that cluster are both present
                shared_cells = data1_M * data2_M
                volume_overlap = np.sum((data1_volumes + data2_volumes) * shared_cells)

                # Sum of where they overlap
                volume_total = np.sum(data1_volumes + data2_volumes)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = (volume_overlap / volume_total) * 100  # volumetric percentage of coverage

            # Filled in 'summaryStatistics' matrix results of percentage overlaps

        usedClusters = set()  # Used to mak sure clusters don't get selected twice
        # Clusters are already sorted by size

        sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
        # go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):
            sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
            # print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)] = 1  # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M == 1)] = 1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    # majority vote
    aggOverlaps = np.nansum(sortedOverlap, axis=0)
    voteOverlaps = np.argmax(aggOverlaps, axis=0)

    # save clusters estimated from the ensemble
    clusters = voteOverlaps

    # compute how uncertain the prediction is
    uncertainty = 1 - np.max(aggOverlaps, axis=0)

    return clusters, uncertainty


@delayed
def nemi_function_volume_dask(df, pack, base_id):
    # if "label_color" in df.columns:
    #     df = df.drop("label_color", axis=1)
    # print(base_id)

    # compute nemi labels and uncertainty
    # st = time.time()
    final_labels, uncertainty = nemi_function_volume(nemi_pack=pack, base_id=base_id)
    # print(time.time() - st)

    # store
    df["final_label"] = final_labels
    df["uncertainty"] = uncertainty * 100
    df = utils.color_code_labels(df, column_name="final_label").rename({"color": "label_color"}, axis=1)
    df.to_csv(f"{prefix}nemi_iteration{base_id}_uncertainty.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print("Usage: python3 script.py <int1> <int2> ... <intN>")
        sys.exit(1)

    try:
        int_list = [int(arg) for arg in sys.argv[1:]]
    except ValueError:
        print("All parameters should be int.")
        sys.exit(1)

    # create dask client
    client = Client(n_workers=1, threads_per_worker=1)

    # load all clustering runs
    pack = []
    for filename in tqdm(glob.glob("output_final/dbscan/uncertainty/UMAP_DBSCAN/umap_dbscan_*.csv")):
        if not "manualkneedrop" in filename:
            i = int(filename.split("/")[-1].lstrip("umap_dbscan_").rstrip(".csv"))
            # load data
            df = pd.read_csv(filename)
            df.label = df.label + 1  # make sure no label is -1 (noise in DBSCAN)
            pack.append([i, df])

    # sort by size and compute volumes
    for i, cl in tqdm(pack):
        clusters = cl.label
        n_clusters = len(clusters.unique())
        hist, _ = np.histogram(clusters, np.arange(n_clusters + 1))
        sorted_clusters = np.argsort(hist)[::-1]  # sort from largest to smallest (if same size, last cluster is taken)
        new_labels = np.full(clusters.shape, np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label
        cl["sorted_label"] = new_labels

        # compute volume
        dlat, dlon = [1, 1]
        depths = np.append(np.sort(cl.LEV_M.unique()), 5000)
        cl.loc[:, "volume"] = cl.apply(compute_volume_alternative, axis=1, args=(dlat, dlon, depths))  # careful with rounding

    prefix = "volume_"
    df = pack[0][1]

    delayed_res = []
    for base_id in tqdm(int_list):
        delayed_res.append(nemi_function_volume_dask(df=df, pack=pack, base_id=base_id))

    res = dask.compute(*delayed_res)
