# adapted from https://github.com/maikejulie/NEMI
import os.path

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
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.graph_objs as go
import plotly.express as px
import matplotlib
from matplotlib.colors import hex2color
from geopy import distance
import math


def compute_area(row, dlat, dlon):
    """ Compute surface area of a grid cell. """
    lat = row["LATITUDE"]
    lon = row["LONGITUDE"]

    R = 6378137.0  # earth radius in meters (WGS84 standard)

    lat_length = 111320  # average meters per degree of latitude
    lon_length = (math.pi / 180) * R * math.cos(math.radians(lat))  # length of 1 degree of longitude in meters at given latitude

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
    next_depth = depths[idx+1]
    ddepth = (next_depth - depth).flatten()[0]

    R = 6378137.0  # earth radius in meters (WGS84 standard)

    lat_length = 111320  # average meters per degree of latitude
    lon_length = (math.pi / 180) * R * math.cos(math.radians(lat))  # length of 1 degree of longitude in meters at given latitude

    volume = dlat * lat_length * dlon * lon_length * ddepth
    return volume


def compute_volume(row, dlat, dlon, depths):
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


def plot_ts_uncertainty(df, color="uncertainty", figsize=(4, 4), xlim=None, save_as=None):
    """ Plot T-S diagram. """
    temp = df.copy()

    # compute necessary parameters
    temp["pressure"] = gsw.p_from_z(-1 * temp["LEV_M"], temp["LATITUDE"])
    temp["abs_salinity"] = gsw.SA_from_SP(temp["P_SALINITY"], temp["pressure"], temp["LONGITUDE"], temp["LATITUDE"])
    temp["cons_temperature"] = gsw.CT_from_pt(temp["abs_salinity"], temp["P_TEMPERATURE"])
    temp["rho"] = gsw.rho(temp["abs_salinity"], temp["cons_temperature"], temp["pressure"])

    # limits
    smin = temp["abs_salinity"].min() - (0.01 * temp["abs_salinity"].min())
    smax = temp["abs_salinity"].max() + (0.01 * temp["abs_salinity"].max())
    tmin = temp["cons_temperature"].min() - (0.1 * temp["cons_temperature"].max())
    tmax = temp["cons_temperature"].max() + (0.1 * temp["cons_temperature"].max())

    if xlim:
        smin = xlim[0] - (0.01 * xlim[0])
        smax = xlim[1] + (0.01 * xlim[1])

    # number of gridcells in the x and y dimensions
    xdim = int(round((smax - smin) / 0.1 + 1, 0))
    ydim = int(round((tmax - tmin) / 0.1 + 1, 0))

    # empty grid
    dens = np.zeros((ydim, xdim))

    # temperature and salinity vectors
    si = np.linspace(1, xdim - 1, xdim) * 0.1 + smin
    ti = np.linspace(1, ydim - 1, ydim) * 0.1 + tmin

    # fill grid with densities
    for j in range(0, int(ydim)):
        for i in range(0, int(xdim)):
            dens[j, i] = gsw.rho(si[i], ti[j], 0)

    # convert to sigma-t
    dens = dens - 1000

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    contours = plt.contour(si, ti, dens, linestyles='dashed', colors='k')
    plt.clabel(contours, fontsize=12, inline=1, fmt='%1.1f')  # label every second level
    ax.scatter(x=temp["abs_salinity"], y=temp["cons_temperature"], s=9, c=temp[color], alpha=1, marker=".",
               vmin=0, vmax=100)  # @todo only this is different?
    ax.set_xlabel('Absolute salinity [g/kg]')
    ax.set_ylabel('Conservative temperature [°C]')
    ax.set_xlim(xlim)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


def coupled_label_plot_uncertainty(df, iteration=0, color_label="uncertainty", suffix="", save=True, save_as=None):
    if save_as is None:
        save_as = [f"nemi_iteration{iteration}_{color_label}_geo{suffix}.png",
                   f"nemi_iteration{iteration}_{color_label}_umap{suffix}.png"
                   f"nemi_iteration{iteration}_{color_label}_boxplot{suffix}.png"]
    temp = df.copy()
    mymap = Basemap(llcrnrlon=temp["LONGITUDE"].min(), llcrnrlat=temp["LATITUDE"].min(),
                    urcrnrlon=temp["LONGITUDE"].max(), urcrnrlat=temp["LATITUDE"].max(), fix_aspect=False)

    # Geospace
    figsize = (6, 6)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    sc_3d = ax.scatter(temp["LONGITUDE"], temp["LATITUDE"], temp["LEV_M"], c=temp[color_label], s=0.5, alpha=1,
                       zorder=4, vmin=0, vmax=100)  # df["predictions"]
    ax.add_collection3d(mymap.drawcoastlines(linewidth=0.5))
    ax.set_box_aspect((np.ptp(temp["LONGITUDE"]), np.ptp(temp["LATITUDE"]),
                       np.ptp(temp["LEV_M"]) / 50))  # aspect ratio is 1:1:1 in data space
    plt.gca().invert_zaxis()
    plt.colorbar(sc_3d, location="bottom", fraction=0.05, pad=0.01, label=color_label.capitalize() + " [%]")
    plt.tight_layout()
    if save:
        plt.savefig(save_as[0])
    plt.show()

    # Embedded space
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    sc_umap = ax.scatter(temp["e0"], temp["e1"], temp["e2"], c=temp[color_label], alpha=0.8, zorder=4,
                         s=1, vmin=0, vmax=100)  # , s=s, alpha=1, zorder=4)
    plt.colorbar(sc_umap, location="bottom", fraction=0.05, pad=0.05, label=color_label.capitalize() + " [%]")
    plt.tight_layout()
    if save:
        plt.savefig(save_as[1])
    plt.show()

    # boxplot
    fig = plt.figure(figsize=(2, 4))
    sns.boxplot(temp[color_label])  # many uncertain points...
    plt.ylabel(color_label.capitalize() + " [%]")
    plt.ylim(0, 100)
    plt.tight_layout()
    if save:
        plt.savefig(save_as[2])
    plt.show(block=True)


def nemi_function(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
        clusters = cl.label
        n_clusters = len(clusters.unique())
        hist, _ = np.histogram(clusters, np.arange(n_clusters + 1))
        sorted_clusters = np.argsort(hist)[::-1]  # sort from largest to smallest (if same size, last cluster is taken)
        new_labels = np.full(clusters.shape, np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label
        cl["sorted_label"] = new_labels

    # list of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(nemi_pack))]
    compare_ids.pop(base_id)

    # identify clusters from the base ensemble member
    base_labels = [x for i, x in nemi_pack if int(i) == int(base_id)][0].sorted_label  # .label.unique())

    # number of clusters
    num_clusters = int(np.max(base_labels) + 1)

    # if not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0])) * np.nan

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [nemi[1].sorted_label for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1 == base_labels)] = 1
            # # Count numer of entries [Why?]
            summaryStats[0, c1] = np.sum(data1_M)

            # go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2 == compare_labels)] = 1

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M * data2_M)

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = (num_overlap / num_total) * 100  # Add percentage of coverage

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
    uncertainty = 1 - np.max(aggOverlaps, axis=0) / len(pack)

    return clusters, uncertainty


def nemi_function_harmonic(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
        clusters = cl.label
        n_clusters = len(clusters.unique())
        hist, _ = np.histogram(clusters, np.arange(n_clusters + 1))
        sorted_clusters = np.argsort(hist)[::-1]  # sort from largest to smallest (if same size, last cluster is taken)
        new_labels = np.full(clusters.shape, np.nan)
        for new_label, old_label in enumerate(sorted_clusters):
            new_labels[clusters == old_label] = new_label
        cl["sorted_label"] = new_labels

    # list of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(nemi_pack))]
    compare_ids.pop(base_id)

    # identify clusters from the base ensemble member
    base_labels = [x for i, x in nemi_pack if int(i) == int(base_id)][0].sorted_label  # .label.unique())

    # number of clusters
    num_clusters = int(np.max(base_labels) + 1)

    # if not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0])) * np.nan

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [nemi[1].sorted_label for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1 == base_labels)] = 1
            # # Count numer of entries [Why?]
            summaryStats[0, c1] = np.sum(data1_M)

            # go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2 == compare_labels)] = 1

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M * data2_M)

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = 2 * (num_overlap / num_total) * 100  # Add percentage of coverage

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
    uncertainty = 1 - np.max(aggOverlaps, axis=0) / len(pack)

    return clusters, uncertainty


def nemi_function_harmonicVolume(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
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

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [(nemi[1].sorted_label, nemi[1].volume) for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels, compare_volumes = dataVector[compare_cnt]
        # compare_labels, compare_areas = dataVector[compare_cnt]

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
                volume_overlap = np.sum(data1_volumes * data2_volumes)

                # Sum of where they overlap
                volume_total = np.sum(data1_volumes + data2_volumes)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = 2*(volume_overlap / volume_total) * 100  # Add percentage of volumetric coverage

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
    uncertainty = 1 - np.max(aggOverlaps, axis=0) / len(pack)

    return clusters, uncertainty


def nemi_function_volumeWeight(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
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

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [(nemi[1].sorted_label, nemi[1].volume) for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels, compare_volumes = dataVector[compare_cnt]
        # compare_labels, compare_areas = dataVector[compare_cnt]

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
                num_overlap = np.sum(data1_M * data2_M)

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = (np.sum(data1_volumes + data2_volumes))/2 * (num_overlap / num_total) * 100  # weight by mean volume

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
    uncertainty = 1 - np.max(aggOverlaps, axis=0) / len(pack)

    return clusters, uncertainty


def nemi_function_areaWeight(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
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
        cl.loc[:, "area"] = cl.apply(compute_area, axis=1, args=(dlat, dlon))

    # list of ensemble members we are comparing to the base
    compare_ids = [i for i in range(len(nemi_pack))]
    compare_ids.pop(base_id)

    # identify clusters from the base ensemble member
    base_labels = [x for i, x in nemi_pack if int(i) == int(base_id)][0].sorted_label
    base_areas = [x for i, x in nemi_pack if int(i) == int(base_id)][0].area

    # number of clusters
    num_clusters = int(np.max(base_labels) + 1)

    # if not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0])) * np.nan

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?
    dataVector = [(nemi[1].sorted_label, nemi[1].area) for id, nemi in enumerate(pack) if id != base_id]

    # loop over ensemble members, not including the base member
    for compare_cnt, compare_id in tqdm(enumerate(compare_ids)):
        # grab clusters of ensemble member
        compare_labels, compare_areas = dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area)
        for c1 in range(max_clusters):
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            # mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1 == base_labels)] = 1
            # # Count numer of entries [Why?]
            summaryStats[0, c1] = np.sum(data1_M)
            # mark areas of the base cluster
            data1_areas = data1_M * base_areas

            # go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int)

                # mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2 == compare_labels)] = 1

                # mark areas of comparison cluster
                data2_areas = data2_M * compare_areas

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M * data2_M)

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)
                summaryStats[c2, c1] = (np.sum(data1_areas + data2_areas))/2 * (num_overlap / num_total) * 100  # weight by mean area
                # summaryStats[c2, c1] = (volume_overlap / volume_total) * 100  # @todo

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
    uncertainty = 1 - np.max(aggOverlaps, axis=0) / len(pack)
    # uncertainty = 1 - np.max(aggOverlaps, axis=0)  # todo

    return clusters, uncertainty


def nemi_function_volume(nemi_pack, base_id: int = 0, max_clusters=None, **kwargs):
    """ From an ensemble of clusterings, compute a final labelling and quantify uncertainty.

    Args:
        base_id (int, optional): index (starting at 0) of ensemble member to use as the base comparison
    """

    base_id = base_id

    # sort by size and compute volumes
    for i, cl in tqdm(nemi_pack):
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

    print(num_clusters, max_clusters)
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


# ------
# load all clustering runs
pack = []
for filename in tqdm(glob.glob("output_final/dbscan/uncertainty/UMAP_DBSCAN/umap_dbscan_*.csv")):
    if not "manualkneedrop" in filename:
        i = int(filename.split("\\")[-1].lstrip("umap_dbscan_").rstrip(".csv"))
        # load data
        df = pd.read_csv(filename)
        df.label = df.label + 1  # make sure no label is -1 (noise in DBSCAN)
        pack.append([i, df])

noise_labels = {"0": 9, "1": 6, "2": 10, "3": 11, "4": 9, "5": 9, "6": 9, "7": 10, "8": 10, "9": 10, "10": 9, "11": 9,
                "12": 10, "13": 9, "14": 10, "15": 10, "22": 10, "42": 10, "67": 10, "99": 10}
done = list(range(58)) + [67]  # 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 42, 99]

for base_id in range(100):
    if base_id not in done:
        if "label_color" in df.columns:
            df = df.drop("label_color", axis=1)
        print(base_id)

        # compute nemi labels and uncertainty
        st = time.time()
        # prefix = ""
        # final_labels, uncertainty = nemi_function(nemi_pack=pack, base_id=base_id)
        # prefix = "harmonic_"
        # final_labels, uncertainty = nemi_function_harmonic(nemi_pack=pack, base_id=base_id)
        # prefix = "harmonicVolume_"
        # final_labels, uncertainty = nemi_function_harmonicVolume(nemi_pack=pack, base_id=base_id)
        # prefix = "volumeWeight_"
        # final_labels, uncertainty = nemi_function_volumeWeight(nemi_pack=pack, base_id=base_id)
        prefix = "areaWeight_"
        final_labels, uncertainty = nemi_function_areaWeight(nemi_pack=pack, base_id=base_id)
        ft = time.time() - st
        print(ft)

        # store
        df["final_label"] = final_labels
        df["uncertainty"] = uncertainty*100
        df = utils.color_code_labels(df, column_name="final_label").rename({"color": "label_color"}, axis=1)
        df.to_csv(f"{prefix}nemi_iteration{base_id}_uncertainty.csv", index=False)

        # plot final labels
        temp = df[df.final_label != noise_labels[str(base_id)]]
        utils.coupled_label_plot(temp, color_label="label_color", save_dir="", suffix="",
                                 save_as=[f"{prefix}nemi_iteration{base_id}_geo",
                                          f"{prefix}nemi_iteration{base_id}_umap"])
        plt.show(block=True)

        # plot uncertainties
        col_name = "uncertainty"
        coupled_label_plot_uncertainty(df, iteration=base_id, color_label=col_name, save=True,
                                       save_as=[f"{prefix}nemi_iteration{base_id}_{col_name}_geo.png",
                                                f"{prefix}nemi_iteration{base_id}_{col_name}_umap.png",
                                                f"{prefix}nemi_iteration{base_id}_{col_name}_boxplot.png"])


# analyse uncertainties and clusterings
dfs = []
dfns = []
prefix = "areaWeight_"
for i, noise_label in noise_labels.items():
    filename = f"{prefix}nemi_iteration{i}_uncertainty.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        df["base_id"] = i
        dfs.append(df)
        dfns.append(df[df["final_label"] != noise_label])
        print(i, df.uncertainty.mean(), df.uncertainty.std())
dfs = pd.concat(dfs)
dfns = pd.concat(dfns)

# ------- compare base_ids
# Number of clusters per base_id
num_clusters = dfs.groupby('base_id')['final_label'].nunique().reset_index()
num_clusters["base_id"] = num_clusters.base_id.astype(int)
num_clusters = num_clusters.sort_values("base_id")
num_clusters["base_id"] = num_clusters.base_id.astype(str)
sns.lineplot(num_clusters, x="base_id", y="final_label")
sns.scatterplot(num_clusters, x="base_id", y="final_label")
plt.xlabel("Base ID")
plt.ylabel("Number of clusters")
plt.tight_layout()
plt.show(block=True)

num_clusters_mean = num_clusters.final_label.mean()
num_clusters_std = num_clusters.final_label.std()
print(f"Mean number of clusters: {num_clusters_mean} +- {num_clusters_std}")
plt.rcParams['figure.figsize'] = (2, 5)
sns.boxplot(num_clusters, zorder=0)
sns.scatterplot({0: num_clusters_mean}, zorder=1)
plt.tight_layout()
plt.show(block=True)

# Uncertainties per base_id (with and without noise)
plt.rcParams['figure.figsize'] = (10, 5)
sns.boxplot(dfs, x="base_id", y="uncertainty", zorder=0)
sns.lineplot(dfs.groupby("base_id")["uncertainty"].mean(), zorder=1, color="red")
sns.scatterplot(dfs.groupby("base_id")["uncertainty"].mean(), zorder=1, color="red")
sns.scatterplot(dfs.groupby("base_id")["uncertainty"].std(), zorder=1, color="orange")
plt.xlabel("Base ID")
plt.ylabel("Uncertainty [%]")
plt.tight_layout()
plt.show(block=True)

plt.rcParams['figure.figsize'] = (10, 5)
sns.boxplot(dfns, x="base_id", y="uncertainty", color="orange", zorder=0)
sns.lineplot(dfns.groupby("base_id")["uncertainty"].mean(), zorder=1, color="red")
sns.scatterplot(dfns.groupby("base_id")["uncertainty"].mean(), zorder=1, color="red")
sns.scatterplot(dfns.groupby("base_id")["uncertainty"].std(), zorder=1, color="orange")
plt.xlabel("Base ID")
plt.ylabel("Uncertainty [%]")
plt.tight_layout()
plt.show(block=True)

# ------- base_id = 67
base_id = "67"
df67 = dfs[dfs.base_id == base_id]  # DBSCAN noise label is 10
dlat, dlon = [1, 1]
depths = np.append(np.sort(df67.LEV_M.unique()), 5000)
df67.loc[:, "volume"] = df67.apply(compute_volume_alternative, axis=1, args=(dlat, dlon, depths))  # careful with rounding

# plot uncertainty per grid cell
sns.lineplot(df67.uncertainty.sort_values().reset_index(drop=True))
plt.ylabel("Uncertainty [%]")
plt.xlabel("Grid cell")
plt.tight_layout()
plt.savefig("iteration67_uncertainty_per_gridcell.png")
plt.show(block=True)

# plot labels for uncertainty < 50%
temp = df67[(df67.final_label != 10) & (df67.uncertainty < 50)]
utils.coupled_label_plot(temp, color_label="label_color", save_dir="", suffix="",
                         save_as=[f"nemi_iteration{base_id}_uncertaintyLess50_geo", f"nemi_iteration{base_id}_uncertaintyLess50_umap"])
plt.show(block=True)

# plot labels for uncertainty >= 50%
temp = df67[(df67.final_label != 10) & (df67.uncertainty >= 50)]
utils.coupled_label_plot(temp, color_label="label_color", save_dir="", suffix="",
                         save_as=[f"nemi_iteration{base_id}_uncertaintyMoreE50_geo", f"nemi_iteration{base_id}_uncertaintyMoreE50_umap"])
plt.show(block=True)

# mean and std uncertainties per cluster
df67_mean_std = df67.groupby("final_label")["uncertainty"].aggregate(["mean", "std"]).reset_index().sort_values("mean")
df67_mean_std["final_label"] = df67_mean_std["final_label"].astype(str)
fig = go.Figure(data=go.Scatter(x=df67_mean_std["final_label"], y=df67_mean_std["mean"], marker_color=df67_mean_std["std"], mode='markers',
                                hovertemplate='Label: %{x}<br>'+'Mean uncertainty: %{y: .2f}%<br>'+'Std uncertainty: %{text}%',
                text=df67_mean_std["std"]))
fig.update_layout(xaxis_title="Final label", yaxis_title="Uncertainty [%]", yaxis_range=[-5, 100])
fig.write_html("mean_uncertainty_per_cluster.html")
fig.show()

# uncertainty boxplots per cluster
x_order = df67_mean_std.final_label.astype(int)
box_plots = []
for cat in x_order:
    box_plots.append(go.Box(y=df67[df67.final_label == cat]["uncertainty"], name=cat,
                            hovertemplate='Label: %{x}<br>'+'Uncertainty: %{y:.2f}%'+'<extra></extra>'))
fig = go.Figure(data=box_plots)
fig.update_layout(xaxis_title="Final label", yaxis_title="Uncertainty [%]", yaxis_range=[-5, 100])
fig.write_html("uncertainty_per_cluster.html")
fig.show()

# ----- Case studies
# medi = df67[df67.final_label.isin([6, 42, 82, 54, 101])]
# lab = df67[df67.final_label.isin([30, 2, 17, 34, 38])]
# deep = df67[df67.final_label.isin([1, 7, 13, 32, 53])]
prefix = "areaWeight_"
base_id = 5
medi = df[df.final_label.isin([7, 42, 88, 98, 75, 101, 0, 11, 21, 29, 45, 51, 134, 209, 273])]
lab = df[df.final_label.isin([2, 17, 25, 27, 31, 33, 50])]
deep = df[df.final_label.isin([6, 14, 32, 49])]

# medi uncertainty, labels, surface, TS
coupled_label_plot_uncertainty(medi, iteration=base_id, color_label="uncertainty",
                               save_as=[f"{prefix}nemi_iteration{base_id}_uncertainty_geo_medi",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_umap_medi",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_boxplot_medi.png"])  # , suffix="_medi")
utils.coupled_label_plot(medi, color_label="label_color", save_dir="", suffix="",
                         save_as=[f"{prefix}nemi_iteration{base_id}_geo_medi",
                                  f"{prefix}nemi_iteration{base_id}_umap_medi",
                                  f"{prefix}nemi_iteration{base_id}_boxplot_medi.png"])
plt.show(block=True)
plot_ts_uncertainty(medi, color="uncertainty", figsize=(4, 4), xlim=None,
                    save_as=f"{prefix}nemi_iteration{base_id}_uncertainty_ts_medi.png")
plt.show(block=True)
utils.plot_ts(medi.rename(columns={"label_color": "color"}), figsize=(4, 4), xlim=None,
              save_as=f"{prefix}nemi_iteration{base_id}_ts_medi.png")
plt.show(block=True)

# lab
coupled_label_plot_uncertainty(lab, iteration=base_id, color_label="uncertainty", suffix="",
                               save_as=[f"{prefix}nemi_iteration{base_id}_uncertainty_geo_lab",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_umap_lab",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_boxplot_lab"])
utils.coupled_label_plot(lab, color_label="color", save_dir="", suffix="",
                         save_as=[f"{prefix}nemi_iteration{base_id}_geo_lab",
                                  f"{prefix}nemi_iteration{base_id}_umap_lab"])
plt.show(block=True)
plot_ts_uncertainty(lab, color="uncertainty", figsize=(4, 4), xlim=None,
                    save_as=f"{prefix}nemi_iteration{base_id}_uncertainty_ts_lab.png")
plt.show(block=True)
utils.plot_ts(lab.rename(columns={"label_color": "color"}), figsize=(4, 4), xlim=None,
              save_as=f"{prefix}nemi_iteration{base_id}_ts_lab.png")
plt.show(block=True)

# deep
coupled_label_plot_uncertainty(deep, iteration=base_id, color_label="uncertainty",
                               save_as=[f"{prefix}nemi_iteration{base_id}_uncertainty_geo_deep",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_umap_deep",
                                        f"{prefix}nemi_iteration{base_id}_uncertainty_boxplot_deep"]
                               )
utils.coupled_label_plot(deep, color_label="color", save_dir="", suffix="",
                         save_as=[f"{prefix}nemi_iteration{base_id}_geo_deep",
                                  f"{prefix}nemi_iteration{base_id}_umap_deep"])
plt.show(block=True)
plot_ts_uncertainty(deep, color="uncertainty", figsize=(4, 4), xlim=None,
                    save_as=f"{prefix}nemi_iteration{base_id}_uncertainty_ts_deep.png")
plt.show(block=True)
utils.plot_ts(deep.rename(columns={"label_color": "color"}), figsize=(4, 4), xlim=None,
              save_as=f"{prefix}nemi_iteration{base_id}_ts_deep.png")
plt.show(block=True)

# plot everything over depth levels , uncertainty <50)
for depth_level in df67.LEV_M.unique():
    temp = df67[(df67.LEV_M == depth_level) & (df67.final_label != 10) & (df67.uncertainty < 50)]
    colors = [hex2color(c) for c in list(temp["label_color"])]
    lat_min = 0
    lat_max = 70
    lon_min = -77
    lon_max = 30
    # plot settings
    factor = 12
    mymap = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max)
    plt.rcParams["figure.figsize"] = ((lon_max - lon_min)/factor, (lat_max - lat_min)/factor)
    x, y = mymap(temp.LONGITUDE, temp.LATITUDE)
    x = list(x)
    y = list(y)
    for i in range(len(x)):
        mymap.scatter(x[i], y[i], color=colors[i], edgecolor='none')  # Adjust 's' for point size
    mymap.drawcoastlines(linewidth=0.5)
    mymap.fillcontinents()
    mymap.drawparallels(np.arange(int(lat_min), int(lat_max), 10), labels=[1, 0, 0, 0])
    mymap.drawmeridians(np.arange(int(lon_min), int(lon_max), 10), labels=[0, 0, 0, 1])
    plt.tight_layout()
    plt.savefig(f"nemi_iteration{base_id}_geo_depth{depth_level}.png")
    plt.show(block=True)


# compare NEMI overlaps
res = []
for base_id in [0, 1, 6, 67]:
    print(base_id)
    a = pd.read_csv(f"nemi_iteration{base_id}_uncertainty.csv")
    a["metric"] = "original"
    dfs = [a]
    hV_file = f"harmonicVolume_nemi_iteration{base_id}_uncertainty.csv"
    vW_file = f"volumeWeight_nemi_iteration{base_id}_uncertainty.csv"
    aW_file = f"areaWeight_nemi_iteration{base_id}_uncertainty.csv"
    h_file = f"harmonic_nemi_iteration{base_id}_uncertainty.csv"
    if os.path.isfile(h_file):
        e = pd.read_csv(h_file)
        e["metric"] = "harmonic"
        dfs.append(e)
    if os.path.isfile(hV_file):
        b = pd.read_csv(hV_file)
        b["metric"] = "harmonicVolume"
        dfs.append(b)
    if os.path.isfile(vW_file):
        c = pd.read_csv(vW_file)
        c["metric"] = "volumeWeight"
        dfs.append(c)
    if os.path.isfile(aW_file):
        d = pd.read_csv(aW_file)
        d["metric"] = "areaWeight"
        dfs.append(d)
    temp = pd.concat(dfs)
    temp["base_id"] = base_id
    res.append(temp)
res = pd.concat(res)
res = res.drop("Unnamed: 0", axis=1)
sizes = res.groupby(["base_id", "final_label", "metric"])["LEV_M"].count().reset_index().rename(columns={"LEV_M": "count"})
res = pd.merge(res, sizes, how="left", on=["base_id", "final_label", "metric"])

# uncertainties
plt.rcParams["figure.figsize"] = (10, 10)
sns.boxplot(res, x="base_id", y="uncertainty", hue="metric")
plt.xlabel("Base ID")
plt.ylabel("Uncertainty [%]")
plt.tight_layout()
plt.show(block=True)

# cluster sizes
plt.rcParams["figure.figsize"] = (10, 10)
sns.boxplot(res, x="base_id", y="count", hue="metric")
plt.xlabel("Base ID")
plt.ylabel("Number of grid cells per clusters")
plt.tight_layout()
plt.show(block=True)

res.groupby(["metric"])["uncertainty"].mean()




noise_labels = {"0": 9, "1": 6, "2": 10, "3": 11, "4": 9, "5": 9, "6": 9, "7": 10, "8": 10, "9": 10, "10": 9, "11": 9,
                "12": 10, "13": 9, "14": 10, "15": 10, "22": 10, "42": 10, "67": 10, "99": 10}
done = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 67]
for base_id in range(100):
    if base_id not in done:
        print(base_id)
        prefix = "areaWeight_"
        file = f"{prefix}nemi_iteration{base_id}_uncertainty.csv"
        if os.path.isfile(file):
            df = pd.read_csv(file).drop(columns=["Unnamed: 0"], axis=1)

            # plot final labels
            temp = df[df.final_label != noise_labels[str(base_id)]]
            utils.coupled_label_plot(temp, color_label="label_color", save_dir="", suffix="",
                                     save_as=[f"{prefix}nemi_iteration{base_id}_geo",
                                              f"{prefix}nemi_iteration{base_id}_umap"])
            plt.show(block=True)

            # plot uncertainties
            col_name = "uncertainty"
            coupled_label_plot_uncertainty(df, iteration=base_id, color_label=col_name, save=True,
                                           save_as=[f"{prefix}nemi_iteration{base_id}_{col_name}_geo.png",
                                                    f"{prefix}nemi_iteration{base_id}_{col_name}_umap.png",
                                                    f"{prefix}nemi_iteration{base_id}_{col_name}_boxplot.png"])

