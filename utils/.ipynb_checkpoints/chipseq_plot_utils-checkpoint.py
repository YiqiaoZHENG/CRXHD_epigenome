import os
import sys
import warnings
import re

import numpy as np
import pandas as pd
import scipy    
from scipy.cluster.hierarchy import fcluster
from scipy import stats
from scipy.stats import mannwhitneyu, normaltest
import fastcluster

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager
import seaborn as sns
#from statannotations.Annotator import Annotator

from utils import specseq_plot_utils

# I. Miscellaneous functions for data processing and plot handling

# heatmap color scheme
div_heat_colors = mpl.colors.LinearSegmentedColormap.from_list(
                        "yq_divergent", [(0, "#CF8B03"), (0.5, "#FFFFFF"), (1, "#08306B")])
single_heat_colors = mpl.colors.LinearSegmentedColormap.from_list(
                        "yq_single", [(0, "#D5DCE6"), (1, "#08306B")])

def palette2hex(mpl_pal):
    # check if requested palette is valid
    if mpl_pal in cm._cmap_registry.keys():
        # from a matplotlib palette, retireve color in hex
        cmap = cm.get_cmap(mpl_pal)
        cmap_hex = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3]  # will return rgba, we take only first 3 so we get rgb
            cmap_hex.append(matplotlib.colors.rgb2hex(rgb))
        return cmap_hex
    else:
        warnings.warn(mpl_pal + " is not a matplotlib palette")

def setup_multiplot(n_plots, n_cols=2, sharex=True, sharey='row', big_dimensions=True):
    """
    Setup a multiplot and hide any superfluous axes that may result.

    Parameters
    ----------
    n_plots : int
        Number of subplots to make
    n_cols : int
        Number of columns in the multiplot. Number of rows is inferred.
    sharex : bool
        Indicate if the x-axis should be shared.
    sharey : bool
        Indicate if the y-axis should be shared.
    big_dimensions : bool
        If True, then the size of the multiplot is the default figure size multiplied by the number of rows/columns.
        If False, then the entire figure is the default figure size.

    Returns
    -------
    fig : figure handle
    ax_list : list-like
        The list returned by plt.subplots(), but any superfluous axes are removed and replaced by None
    """
    n_rows = int(np.ceil(n_plots / n_cols))
    row_size, col_size = mpl.rcParams["figure.figsize"]

    if big_dimensions:
        # A bit counter-intuitive...the SIZE of the row is the width, which depends on the number of columns
        row_size *= n_cols
        col_size *= n_rows

    fig, ax_list = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(row_size, col_size), sharex=sharex, sharey=sharey)

    # The index corresponding to n_plots is the first subplot to be hidden
    for i in range(ax_list.size):
        coords = np.unravel_index(i, ax_list.shape)
        ax = ax_list[coords]
        if i >= n_plots:
            ax.remove()
            ax_list[coords] = None

    return fig, ax_list

def read_profile_table(filename, rowgroup="region"):
    # load the regions by score profile gz matrix
    profile_df = pd.read_csv(filename, sep="\t", header=1).dropna(axis="columns", how="all", inplace=False)
    if rowgroup == "score:": # for making score by region profile plot
        profile_df = profile_df.rename(columns = {"bins": "region", "Unnamed: 1": "score"}, errors="raise")
    else: # for making region by score profile plot, default
        profile_df = profile_df.rename(columns = {"bins": "score", "Unnamed: 1": "region"}, errors="raise")

    # retreive unique score and region names
    score_names = profile_df.score.unique()
    region_names = profile_df.region.unique()

    # get bin size from gz matrix, it is stored in the first line
    bin_label = pd.read_csv(filename, sep="\t", header=None).iloc[0, 0:len(profile_df.columns)]
    # get the label and positions
    bin_label = bin_label[bin_label.notna()]

    bin_label_pos = bin_label.index[1:]-2
    bin_label_name = bin_label.values[1:]

    return (profile_df, score_names, region_names, bin_label_pos, bin_label_name)

# II. functions for making clustered heatmaps
def chip_intensity_heatmap(data,  hm_title=None, hm_xlabel=None, cb_title=None, cmap=div_heat_colors, paramdict=None):
    # default seaborn clustermap parameters
    default_params = {
    # data, normalization, clustering
    'z_score': 0, # z normalized by row
    'metric': "euclidean", # plot with euclidean distance
    'method': "complete", # linkage method to use for calculating clusters: Farthest Point Algorithm/Voor Hees Algorithm
    'row_cluster': True,
    'col_cluster': False,

    # dendrogram, colorbar, colormap
    'cbar_pos': (1, .3, .03, .4), #tuple of (left, bottom, width, height),
    'robust': True,
    'center': 0.0,
    'cmap': cmap,
    'cbar_kws': { # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
            'orientation': "vertical",
            'ticklocation': "right",
            'pad': 0.08 # padding between the color bar and new image axes
            },

    # figsize, axes ticks, axes lables
    'yticklabels': False,
    'figsize': mpl.rcParams["figure.figsize"]
    }

    # add additional parameters if specified
    if paramdict:
        #plot_params = default_params | paramdict # for py3.9+
        plot_params = {**default_params, **paramdict} # py3.5+
    else:
        plot_params = default_params

    # cluster data and make heatmap
    cg = sns.clustermap(data=data.copy(), **plot_params)
    # omit the dendrogram
    cg.ax_row_dendrogram.set_visible(False)

    # get the heatmap axes
    ax = cg.ax_heatmap
    # heatmap title
    if hm_title:
        ax.set_title(hm_title)
    # heatmap axes labels
    if hm_xlabel:
        ax.set_xlabel(hm_xlabel, fontsize=mpl.rcParams["axes.titlesize"])
    ax.get_yaxis().set_visible(False) # ax.set_ylabel("")
    # heatmap axis tick labels
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=mpl.rcParams["axes.titlesize"])

    # name the color bar
    if cb_title:
        cg.ax_cbar.set_title(cb_title, fontsize=mpl.rcParams["axes.labelsize"], pad=8.0)

    return cg

### check cluster definition
def nclust_heatmap(data, clust_col, nclust=1, cmap=div_heat_colors, paramdict=None):

    # first check if dataframe has all clust_col
    for col in clust_col:
        if col not in data.columns:
            warnings.warn("Column name " + col + " not found in dataframe")

    # default seaborn clustermap parameters
    default_params = {
    # data, normalization, clustering
    'z_score': 0, # z normalized by row
    'metric': "euclidean", # plot with euclidean distance
    'method': "complete", # linkage method to use for calculating clusters: Farthest Point Algorithm/Voor Hees Algorithm
    'row_cluster': True,
    'col_cluster': False,

    # dendrogram, colorbar, colormap
    'cbar_pos': (1, .3, .03, .4), #tuple of (left, bottom, width, height),
    'robust': True,
    'center': 0.0,
    'cmap': cmap,
    'cbar_kws': { # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
            'orientation': "vertical",
            'ticklocation': "right",
            'pad': 0.08 # padding between the color bar and new image axes
            },

    # figsize, axes ticks, axes lables
    'yticklabels': False,
    'figsize': mpl.rcParams["figure.figsize"]
    }

    # add additional parameters if specified
    if paramdict:
        plot_params = default_params | paramdict # for py3.9+
    else:
        plot_params = default_params
    
    data = data.copy()
    # calculate linkage matrix
    row_linkage = fastcluster.linkage(data.loc[:,clust_col], method='complete', metric='euclidean', preserve_input='True')

    # retrieve flat clusters
    row_cluster = scipy.cluster.hierarchy.fcluster(row_linkage, nclust, criterion='maxclust')
    # assign cluster number to peak
    data["row_cluster"] = row_cluster

    cg_list = []

    # make a small heatmap for each cluster defined
    for i in range(1,nclust+1,1):
        data_to_plot = data.loc[data.row_cluster == i,clust_col].copy()
        print("number of peaks in cluster " + str(i) + " " + str(len(data_to_plot)))
        # make heatmap
        cg = sns.clustermap(data = data_to_plot, **plot_params)
        # omit the dendrogram
        cg.ax_row_dendrogram.set_visible(False)
        # name heatmap with cluster number
        cg.ax_heatmap.set_title("cluster " + str(i))
        # remove heatmap yaxis labels
        cg.ax_heatmap.get_yaxis().set_visible(False)

        cg_list.append(cg)
    
    return data, cg_list

def parse_clustered_peakset(df, cluster_col, prefix):
    df = df.copy()
    for name in df[cluster_col].unique():
        small_df = df.loc[df.row_cluster == name, ["seqnames", "start", "end"]]
        small_df.to_csv(os.path.join(prefix, name+"_regions.bed"), sep="\t", header=False, index=False)


# III. functions for making profile line plots

def profile_line(data, score_names, region_names,  bin_label_pos, bin_label_name, cmap, sharex=True, sharey="row", temp_params=None):
    if temp_params :
        default_params = {k:mpl.rcParams[k] for k in temp_params.keys()}
        plt.rcParams.update(temp_params)

    # set up plots
    fig, ax_list = setup_multiplot(n_plots=len(score_names)*len(region_names), n_cols=len(score_names), sharex=sharex, sharey=sharey, big_dimensions=True)

    for i in range(len(region_names)):
        region = region_names[i]
        # retrieve data
        small_data = data.loc[data.region == region,:].copy()
        for j in range(len(score_names)):
            # retreive axes
            ax = ax_list[i, j]

            score = score_names[j]
            ax.plot(small_data.columns[2:], small_data.iloc[j, 2:], color=cmap[j], label=score)
            
            # format axis labels and ticks
            ax.set_xticks(bin_label_pos)
            ax.set_xticklabels(bin_label_name)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            # display axis label only on the edge
            ss = ax.get_subplotspec()
            if ss.is_first_row() :
                ax.set_title(score)
            if ss.is_first_col():
                ax.set_ylabel(region, labelpad=8.0)
    
    # restore default rmParams
    plt.rcParams.update(default_params)
    
    return fig, ax_list


def stacked_line(data, score_names, region_names,  bin_label_pos, bin_label_name, cmap, show_lg=True, sharex=True, sharey=True, temp_params=None):
    if temp_params:
        default_params = {k:mpl.rcParams[k] for k in temp_params.keys()}
        plt.rcParams.update(temp_params)
    
    # set up plots
    fig, ax_list = setup_multiplot(n_plots=len(score_names), n_cols=len(score_names), sharex=sharex, sharey=sharey, big_dimensions=True)
    ax_list = ax_list.flatten()

    
    for j in range(len(score_names)):
        # retreive axes
        ax = ax_list[j]
        score = score_names[j]
        for i in range(len(region_names)):
            # retrieve data
            region = region_names[i]
            small_data = data.loc[data.region == region,:].copy()

            # add a line to the plot
            ax.plot(small_data.columns[2:], small_data.iloc[j, 2:], color=cmap[i], label=region)
            
        # format axis labels and ticks
        ax.set_xticks(bin_label_pos)
        ax.set_xticklabels(bin_label_name)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # display axis label on top
        ss = ax.get_subplotspec()
        if ss.is_first_row():
            ax.set_title(score)
    
    # add legend
    if show_lg:
        leg = ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0.8)

    # restore default rmParams
    plt.rcParams.update(default_params)
    
    return fig, ax_list

def make_profile_plot(filename, plot_type="line", rowgroup="region", bin_label_name = ["-2.0kb", "summit", "2.0kb"], 
                        cmap="tab10", sharex=True, sharey='row', show_lg=False, temp_params=None, 
                        fig_name=None, figure_dir=None, sub_dir="chip.profile"):
    # parse the profile matrix
    profile_df, score_names, region_names, bin_label_pos, temp_bin_label_name = read_profile_table(filename, rowgroup=rowgroup)
    if not bin_label_name:
        bin_label_name = temp_bin_label_name
        del(temp_bin_label_name)

    print("All scores: " + ", ".join(score_names))
    print("All regions: " + ", ".join(region_names))

    # retrieve matplotlib color in hex
    cmap_hex = palette2hex(cmap)

    # format the plot parameters
    plot_params={
        'data': profile_df,
        'score_names': score_names,
        'region_names': region_names,
        'bin_label_pos': bin_label_pos,
        'bin_label_name': bin_label_name,
        'cmap': cmap_hex,
        'temp_params': temp_params,
        'sharex': sharex,
        'sharey': sharey
    }
    
    if plot_type == "stacked_line" or plot_type == "stacked": # plot stacked lines
        fig, ax_list = stacked_line(show_lg=show_lg, **plot_params)
    else: # plot each score x region sample separatly
        fig, ax_list = profile_line(**plot_params)

    # infer plot name if not specified
    if fig_name:
        fig_name = os.path.join(figure_dir, sub_dir, fig_name)
    else:
        fig_name = os.path.join(figure_dir, sub_dir, ".".join([os.path.split(filename)[1].split(".")[0], plot_type, "by" + rowgroup]))
        if sharey in ["row", "column"]:
            fig_name += ".y" + sharey
    
    return profile_df, fig, fig_name



