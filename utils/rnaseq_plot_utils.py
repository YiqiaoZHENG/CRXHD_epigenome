
import os
import sys
import warnings
import re
import itertools

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy
from scipy import stats
from scipy.stats import mannwhitneyu, normaltest
import fastcluster

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager
import seaborn as sns
#from statannotations.Annotator import Annotator

from utils import specseq_plot_utils

# I. Functions for making grouped box/strip/violin plots

def annotate_test_stats(data, x_column, y_column, morder, figax, test_pairs, test='Mann-Whitney', test_format="star"):
    # retrieve ax to add annotation
    fig, ax = figax

    valid_name = [x.get_text() for x in ax.get_xticklabels()]

    pairs = []
    # check if all pairs are valid
    for pair in test_pairs:
        if pair[0] in valid_name and pair[1] in valid_name:
            pairs.append(pair)
    
    if len(pairs) >= 1:
        annotator = Annotator(ax, pairs, data=data, x=x_column, y=y_column, order=morder)
        # adjust for multiple tests
        # ref: https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        # test value should be a StatTest instance or one of the following strings: t-test_ind, t-test_welch, t-test_paired, Mann-Whitney, Mann-Whitney-gt, Mann-Whitney-ls, Levene, Wilcoxon, Kruskal.
        annotator.configure(test=test, text_format=test_format, comparisons_correction="fdr_bh", verbose=2, line_width=mpl.rcParams["lines.linewidth"], line_height=0.0)
        annotator.apply_and_annotate()
    
    return fig, ax


def box_by_category(data, x_column, y_column, morder, mpal, annot_pairs, annot_bool=True, xlabel=None, ylabel=None, patch_artist=False, figax=None):

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    data = data.copy()

    # check if all keys exist
    v_order = [od for od in morder if od in data[x_column].unique()]
    v_color = pd.Series({k:mpal[k] for k in v_order})

    data = data[data[x_column].isin(v_order)]

    
    if patch_artist:
        # fill box with color
        sns.boxplot(x=x_column, y=y_column, data=data, showfliers = False, ax=ax, order=v_order, palette=v_color)
    else:
        # remove fill color
        boxplot_kwargs = {'boxprops' : {'edgecolor': 'k', 'facecolor': 'none'}}
        # make the plot
        sns.boxplot(x=x_column, y=y_column, data=data, showfliers = False, ax=ax, order=v_order, **boxplot_kwargs)
        
        # color the lines of each box
        for i, artist in enumerate(ax.artists):
            col = v_color[i]
            # This sets the color for the main box
            artist.set_edgecolor(col)
            # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
            # if outlier is removed only 5 Line2D objects
            # automatically get the object number
            n_line2d_objs =  int(len(ax.lines)/len(v_order))
            # Loop over them here, and use the same colour as above
            for j in range(i*n_line2d_objs,i*n_line2d_objs+n_line2d_objs):
                line = ax.lines[j]
                line.set_color(col)
                line.set_mfc(col)
                line.set_mec(col)

    # set axis labels
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
    
    # add annotation
    if annot_bool and len(annot_pairs)>0 and len(v_order)>1:
        fig, ax = annotate_test_stats(data, x_column, y_column, v_order, figax=(fig, ax), test_pairs=annot_pairs)

    return fig, ax


def strip_by_category(data, x_column, y_column, morder,  mpal, add_mean=False, showmean_th=15, mean_wd=.25, markersize=8, xlabel=None, ylabel=None, ylimits=None, annot_pairs=None, annot_bool=True, figax=None):
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    data = data.copy()

    idx_counts = data[x_column].value_counts()
    # check if all keys exist and keep only groups with at least five datapoints
    v_order = [od for od in morder if od in data[x_column].unique() and idx_counts[od] >=5]
    v_color = pd.Series({k:mpal[k] for k in v_order})
    data = data[data[x_column].isin(v_order)]

    # check if all keys exist
    #v_order = [od for od in morder if od in data[x_column].unique()]
    #v_color = pd.Series({k:mpal[k] for k in v_order})
    
    sns.stripplot(x=x_column, y=y_column, data=data, ax=ax, order=v_order, palette=v_color, size=markersize, zorder=1)
    
    # add mean line
    if add_mean:
        if not mean_wd:
            mean_wd=.25
        hline_paras = dict(linestyle='-', color = 'black', alpha=0.6)
        means = data.groupby(x_column, sort=False)[y_column].mean()
        _ = [ax.hlines(means[od], i-mean_wd, i+mean_wd, **hline_paras, zorder=2) for i,od in enumerate(v_order) if idx_counts[od] >= showmean_th]

    # set axis labels
    if xlabel:
        ax.set(xlabel=xlabel)
    if ylabel:
        ax.set(ylabel=ylabel)

    if ylimits:
        ax.set_ylim(ylimits)

    # add annotation if all conditions met
    if annot_bool and len(annot_pairs)>0 and len(v_order)>1:
        idx_counts = data[x_column].value_counts()
        # only annotate pairs with at least xx datapoint available
        pairs = [pair for pair in annot_pairs if pair[0] in idx_counts.index and pair[1] in idx_counts.index and idx_counts[pair[0]]>=showmean_th and idx_counts[pair[1]]>=showmean_th]
        fig, ax = annotate_test_stats(data, x_column, y_column, v_order, figax=(fig, ax), test_pairs=pairs)

    return fig, ax


def violin_by_category(data, x_column, y_column, morder, mpal, xlabel=None, ylabel=None, figax=None):

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    data = data.copy()

    # check if all keys exist
    v_order = [od for od in morder if od in data[x_column].unique()]
    v_color = pd.Series({k:mpal[k] for k in v_order})

    data = data[data[x_column].isin(v_order)]

    sns.violinplot(x=x_column, y=y_column, data=data, cut=0, ax=ax, order= v_order, palette=v_color)

    # set axis labels
    if xlabel:
        ax.set(xlabel=xlabel)
    if ylabel:
        ax.set(ylabel=ylabel)

    return fig, ax


def rnalfc_bychip_box(data_to_plot, colors=["#1A1A1A" ,"#A9A9A9", "#CF8B03", "#08306B"], annot_bool=True, figax=None):
    
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(4, 4, sharex=False, sharey=False) # note, do not sharey, it will use the smallest y limits
    ax_list = ax_list.flatten()

    morder = ["No Peak", "No Change", "Lost", "Gained"]
    mpal = pd.Series({a:b for a,b in zip(morder, colors)})

    boxplot_parameters={
        'x_column': "chip_group",
        'y_column': data_to_plot.columns[5],
        'morder': morder,
        'mpal': mpal,
        'xlabel': "ChIP group",
        'ylabel': "RNA log2FC[mut/WT]",
        'annot_bool': annot_bool,
        'annot_pairs': [("No Peak", "No Change"), ("No Peak","Lost"), ("No Peak", "Gained"), ("No Change", "Lost"), ("No Change", "Gained"), ("Lost", "Gained")]
    }

    # boxplot 
    fig, ax_list[0] = box_by_category(data=data_to_plot[(data_to_plot["CRE_group"]=="Promoter") | (data_to_plot["CRE_group"]=="Distal Enhancer")], **boxplot_parameters, figax=(fig, ax_list[0]))
    fig, ax_list[1] = box_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Promoter"], **boxplot_parameters, figax=(fig, ax_list[1]))
    fig, ax_list[2] = box_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Distal Enhancer"], **boxplot_parameters, figax=(fig, ax_list[2]))
    fig, ax_list[3] = box_by_category(data=data_to_plot, **boxplot_parameters, figax=(fig, ax_list[3]))

    # name each panel
    ax_list[0].set_title("All CREs")
    ax_list[1].set_title("Promoter")
    ax_list[2].set_title("Distal Enhancer")
    ax_list[3].set_title("All Genes")

    # adjust for y axes, make them the same
    yaxis_limits = list(zip(*[ax.get_ylim() for ax in ax_list]))
    bottom = min(yaxis_limits[0])
    top = max(yaxis_limits[1])

    for ax in fig.get_axes():
        ax.set_ylim(bottom, top)
        ss = ax.get_subplotspec()
        if not ss.is_first_col():
            #ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
            ax.set_ylabel("") # high only the title

    fig.tight_layout()

    return fig


def rnalfc_bychip_strip(data_to_plot, colors=["#1A1A1A" ,"#A9A9A9", "#CF8B03", "#08306B"], annot_bool=True, figax=None):
    
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(4, 4, sharex=False, sharey=False) # note, do not sharey, it will use the smallest y limits
    ax_list = ax_list.flatten()

    morder = ["No Peak", "No Change", "Lost", "Gained"]
    mpal = pd.Series({a:b for a,b in zip(morder, colors)})

    stripplot_parameters = {
        'x_column': "chip_group",
        'y_column': data_to_plot.columns[5],
        'morder': morder,
        'mpal': mpal,
        'xlabel': "ChIP group",
        'ylabel': "RNA log2FC[mut/WT]",
        'annot_bool': annot_bool,
        'annot_pairs': [("No Peak", "No Change"), ("No Peak","Lost"), ("No Peak", "Gained"), ("No Change", "Lost"), ("No Change", "Gained"), ("Lost", "Gained")]
    }

    # stripplot
    fig, ax_list[0] = strip_by_category(data=data_to_plot[(data_to_plot["CRE_group"]=="Promoter") | (data_to_plot["CRE_group"]=="Distal Enhancer")], **stripplot_parameters, figax=(fig, ax_list[0]))
    fig, ax_list[1] = strip_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Promoter"], **stripplot_parameters, figax=(fig, ax_list[1]))
    fig, ax_list[2] = strip_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Distal Enhancer"], **stripplot_parameters, figax=(fig, ax_list[2]))
    fig, ax_list[3] = strip_by_category(data=data_to_plot, **stripplot_parameters, figax=(fig, ax_list[3]))

    # name each panel
    ax_list[0].set_title("All CREs")
    ax_list[1].set_title("Promoter")
    ax_list[2].set_title("Distal Enhancer")
    ax_list[3].set_title("All Genes")

    # adjust for y axes, make them the same
    yaxis_limits = list(zip(*[ax.get_ylim() for ax in ax_list]))
    bottom = min(yaxis_limits[0])
    top = max(yaxis_limits[1])

    for ax in fig.get_axes():
        ax.set_ylim(bottom, top)
        ss = ax.get_subplotspec()
        if not ss.is_first_col():
            #ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
            ax.set_ylabel("") # high only the title

    fig.tight_layout()

    return fig



def rnalfc_bychip_violin(data_to_plot, colors=["#1A1A1A" ,"#A9A9A9", "#CF8B03", "#08306B"], annot_bool=False, figax=None):
    
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(4, 4, sharex=False, sharey=False) # note, do not sharey, it will use the smallest y limits
    ax_list = ax_list.flatten()

    morder = ["No Peak", "No Change", "Lost", "Gained"]
    mpal = pd.Series({a:b for a,b in zip(morder, colors)})

    violinplot_parameters={
        'x_column': "chip_group",
        'y_column': data_to_plot.columns[5],
        'morder': morder,
        'mpal': mpal,
        'xlabel': "ChIP group",
        'ylabel': "RNA log2FC[mut/WT]"
    }

    # make violinplot
    fig, ax_list[0] = violin_by_category(data=data_to_plot[(data_to_plot["CRE_group"]=="Promoter") | (data_to_plot["CRE_group"]=="Distal Enhancer")], **violinplot_parameters, figax=(fig, ax_list[0]))
    fig, ax_list[1] = violin_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Promoter"], **violinplot_parameters, figax=(fig, ax_list[1]))
    fig, ax_list[2] = violin_by_category(data=data_to_plot[data_to_plot["CRE_group"]=="Distal Enhancer"], **violinplot_parameters, figax=(fig, ax_list[2]))
    fig, ax_list[3] = violin_by_category(data=data_to_plot, **violinplot_parameters, figax=(fig, ax_list[3]))

    # name each panel
    ax_list[0].set_title("All CREs")
    ax_list[1].set_title("Promoter")
    ax_list[2].set_title("Distal Enhancer")
    ax_list[3].set_title("All Genes")

    # adjust for y axes, make them the same
    yaxis_limits = list(zip(*[ax.get_ylim() for ax in ax_list]))
    bottom = min(yaxis_limits[0])
    top = max(yaxis_limits[1])

    for ax in fig.get_axes():
        ax.set_ylim(bottom, top)
        ss = ax.get_subplotspec()
        if not ss.is_first_col():
            #ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
            ax.set_ylabel("") # high only the title

    fig.tight_layout()

    return fig


# II. Functions for making grouped ran volcano plot

def annotate_gene_on_scatter(data, x_col, y_col, annot_list, figax):
    fig, ax = figax

    all_genes = data.gene.tolist()

    # first check if data contains genes in the annotation list
    for gene in annot_list:
        if gene in all_genes:
            
            # retrieve x and y positions
            x = data.loc[data["gene"]==gene, x_col].values[0]
            y = data.loc[data["gene"]==gene, y_col].values[0]

            ax.annotate(gene, # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='left', # horizontal alignment can be left, right or center
                        clip_on=True) # clip to the axes bounding box
            ax.scatter(data=data, x=x, y=y, facecolor="#08306B", edgecolor="none", s=14)
            
    return fig, ax


def scatter_by_category(data, x, y, xlabel=None, ylabel=None, lr=False, figax=None, annot_bool=False, annot_list=None, **kwargs):

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    sns.scatterplot(data=data, x=x, y=y, **kwargs, ax=ax)

    if lr: 
        # linear regression
        slope, intercept, _, _, _ = stats.linregress(data[x], data[y])
        # calculate Pearson correlation coefficient
        pearson_corr,_ = stats.pearsonr(data[x], data[y])
        # calculate Spearman rank correlation coefficient
        spearman_corr,_ = stats.spearmanr(data[x], data[y])

        # format disply text
        text = (f'r = {pearson_corr:.3f}', f'\u03C1 = {spearman_corr:.3f}', slope, intercept)
    else:
        text=""

    # annotate selected genes
    if annot_bool and len(annot_list)>1:
        fig, ax = annotate_gene_on_scatter(data=data[data["rna_group"]!="No Change"], x_col=x, y_col=y, annot_list=annot_list, figax=(fig, ax))
    
    # set axis labels
    if xlabel:
        ax.set(xlabel=xlabel)
    else:
        ax.set_xlabel("")
    if ylabel:
        ax.set(ylabel=ylabel)
    else:
        ax.set_ylabel("")

    return fig, ax, text


def rnalfc_scatter(data_to_plot, simplify=False, colors=["#A9A9A9", "#CF8B03", "#08306B"], markersize=14, annot_list=None, annot_bool=False, figax=(None)):
    
    # retreive rna data columns
    rna_lfc_col = data_to_plot.columns[5]
    rna_padj_col = data_to_plot.columns[6]
    
    # calculate the -log10FDR
    data_to_plot.loc[:,"log10.fdr"] = -np.log10(data_to_plot[rna_padj_col])

    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(5, 5, sharex=False, sharey=True)
    ax_list = ax_list.flatten()

    if simplify:
        morder = ["No Change", "Diff. Exp."]
        mpal = {a:b for a,b in zip(morder, colors[0:2])}
    else:
        morder = ["No Change", "Lost", "Gained"]
        mpal = {a:b for a,b in zip(morder, colors)}


    scatter_paras={
        'x': rna_lfc_col,
        'y': "log10.fdr",
        'ylabel': "-$log_{10}$(FDR)",
        'hue': "rna_group",
        'palette': mpal,
        'legend': False,
        's': markersize,
        'edgecolor': "none",
        'annot_list': annot_list,
        'annot_bool': annot_bool,
    }

    fig, ax_list[0], _ = scatter_by_category(data=data_to_plot, **scatter_paras, figax=(fig,ax_list[0]))
    fig, ax_list[1], _ = scatter_by_category(data=data_to_plot[data_to_plot["chip_group"]=="No Peak"], **scatter_paras, figax=(fig,ax_list[1]))
    fig, ax_list[2], _ = scatter_by_category(data=data_to_plot[data_to_plot["chip_group"]=="No Change"], **scatter_paras, figax=(fig,ax_list[2]))
    fig, ax_list[3], _ = scatter_by_category(data=data_to_plot[data_to_plot["chip_group"]=="Lost"], **scatter_paras, figax=(fig,ax_list[3]))
    fig, ax_list[4], _ = scatter_by_category(data=data_to_plot[data_to_plot["chip_group"]=="Gained"], **scatter_paras, figax=(fig,ax_list[4]))
    
    ax_list[0].set_title("All Genes")
    ax_list[1].set_title("No Nearby Peak")
    ax_list[2].set_title("No Change Peak")
    ax_list[3].set_title("Peak Decreased")
    ax_list[4].set_title("Peak Increased")

    # adjust for x axes, make them the same and symmetric, and concentrated a little bit
    xaxis_limits = [abs(x) for sublist in list(zip(*[ax.get_xlim() for ax in ax_list])) for x in sublist]
    limit = round(max(xaxis_limits)*1.25)

    for ax in ax_list:
        ax.set_xlim(-limit, limit)
        # add reference lines
        bottom, top = ax.get_ylim()
        ax.set_ylim(0, top)
        #ax.vlines(x=0, ymin=0, ymax=top, linestyle='--', color = 'black', alpha=0.4)
        ax.hlines(y=5, xmin=-limit, xmax=limit, linestyle='--', color = 'black', alpha=0.4)
        # remove y axis labels for subplots to the right
        ss = ax.get_subplotspec()
        if not ss.is_first_col():
            ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
            #ax.set_ylabel("") # high only the title
    
    return fig, ax_list

def chip_rna_scatter(data_to_plot, simplify=False, colors=["#A9A9A9", "#CF8B03", "#08306B"], annot_list=None, annot_bool=False, figax=None):

    data_to_plot = data_to_plot.copy()

    # retreive data columns
    chip_lfc_col = data_to_plot.columns[3]
    chip_fdr_col = data_to_plot.columns[4]
    rna_lfc_col = data_to_plot.columns[5]
    rna_padj_col = data_to_plot.columns[6]

    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    if simplify:
        morder = ["No Change", "Diff. Exp."]
        mpal = {a:b for a,b in zip(morder, colors[0:2])}
    else:
        morder = ["No Change", "Lost", "Gained"]
        mpal = {a:b for a,b in zip(morder, colors)}

    scatter_paras={
        'x': chip_lfc_col,
        'y': rna_lfc_col,
        'xlabel': "ChIP log2FC[mut/WT]",
        'ylabel': "RNA log2FC[mut/WT]",
        'hue': "rna_group",
        'palette': mpal,
        'hue_order': morder[::-1], # order G > L > N
        'legend': False,
        's': 20,
        'edgecolor': "none"
    }

    fig, ax, _ = scatter_by_category(data=data_to_plot.loc[lambda df: df["CRE_group"] != "No CRE",:], **scatter_paras, figax=(fig, ax))

    # adjust for axes, make them symmetric
    ylimit = round(max(abs(y) for y in ax.get_ylim())*1.25)
    xlimit = round(max(abs(x) for x in ax.get_xlim())*1.25)
    ax.set_ylim(-ylimit, ylimit)
    ax.set_xlim(-xlimit, xlimit)

    ax.vlines(x=0, ymin=-ylimit, ymax=ylimit, linestyle='--', color = 'grey', alpha=0.4)
    ax.hlines(y=0, xmin=-xlimit, xmax=xlimit, linestyle='--', color = 'grey', alpha=0.4)
    
    return fig


# III. Functions for making fraction bar plots

def get_category_count(summary_ser, col_index, row_index):
    if (row_index, col_index) in  summary_ser.index:
        return summary_ser[(row_index, col_index)]
    else:
        return 0


def summary_stacked_barplot(summary_df, index_order, col_order, palette, groupby='row', ratio=True, figax=None):
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()

    summary_df = summary_df.reindex(index_order)[col_order].copy()
    df = summary_df

    if groupby == 'column':
        # index as x axis
        ylabel = "Number of genes"
        if ratio:
            df = summary_df.div(summary_df.sum(axis=0), axis=1) # divide column by column sum
            ylabel = "Percentage of genes"

        # Initialize the bottom at zero for the first set of bars.
        bottom = np.zeros(len(df.columns))

        # Plot each layer of the bar, adding each bar to the "bottom" so
        # the next bar starts higher.
        for i, col in enumerate(df.index):
            sns.barplot(x=df.columns, y=df.loc[col,:], bottom=bottom, label=col, color=palette[col], ax=ax)
            bottom += np.array(df.loc[col,:])

        ax.set(xlabel="CRX binding intensity change", ylabel=ylabel)
        ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', ncol=1, title="RNA Change")
    
    else:
        # columns as x axis
        ylabel = "Number of peaks"
        if ratio:
            df = summary_df.div(summary_df.sum(axis=1), axis=0) # divide row by row sum
            ylabel = "Percentage of peaks"
            
        # Initialize the bottom at zero for the first set of bars.
        bottom = np.zeros(len(df.index))

        # Plot each layer of the bar, adding each bar to the "bottom" so
        # the next bar starts higher.
        for i, col in enumerate(df.columns):
            sns.barplot(x=df.index, y=df[col], bottom=bottom, label=col, color=palette[col], ax=ax)
            bottom += np.array(df[col])

        ax.set(xlabel="Gene expression change", ylabel=ylabel)
        ax.legend(bbox_to_anchor=(1.01, 0), loc='lower left', ncol=1, title="Binding Change")

    # hide some spines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    return fig, ax


def single_stacked_bar(ser, index_order, mpal, orient="h", ratio=True, label=None, figax=None):
    if figax:
        fig, ax = figax
    else:
        if orient == "v":
            fig, ax = plt.subplots(figsize=(.4,4))
        else:
            fig, ax = plt.subplots(figsize=(4,.4))

    if ratio:
        ser = ser.div(ser.sum()) # get ratio
    
    # clean up the order
    v_order = [od for od in index_order if od in ser.index.unique()]
    ser = ser.reindex(v_order).copy()
    
    if orient == "v":
        # Initialize the bottom at zero for the first set of bars.
        bottom = 0.0
        x_pos = np.arange(len([ser.name]))

        # Plot each layer of the bar, adding each bar to the "bottom" so
        # the next bar starts higher.
        for i, idx in enumerate(v_order):
            sns.barplot(x=x_pos, y=ser[idx], bottom=bottom, label=idx, color=mpal[idx], ax=ax)
            bottom += ser[idx]
        
        if label:
            ax.set(ylabel=label)
        else:
            ax.set(ylabel="")

        # remove all spines and axes except the one with meaningful numbers
        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.xaxis.set_visible(False)

    if orient == "h":
        # Initialize the left at zero for the first set of bars.
        left = 0.0
        y_pos = np.arange(len([ser.name]))

        ax.invert_yaxis()

        # Plot each layer of the bar, adding each bar to the "left" so
        # the next bar starts to the right.
        for i,idx in enumerate(v_order):
            ax.barh(y=y_pos, width=ser[idx], label=idx, left=left, color=mpal[idx])
            left += ser[idx]

        if label:
            ax.set(xlabel=label)
        else:
            ax.set(xlabel="")

        ax.spines.left.set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig, ax


def barplot_by_category(ser, index_order, mpal, orient="h", ratio=True, label=None, figax=None):
    ser = ser.copy()

    if figax:
        fig, ax = figax
    else:
        if orient == "v":
            fig, ax = plt.subplots(figsize=(1,4))
        else:
            fig, ax = plt.subplots(figsize=(4,1))

    if ratio:
        ser = ser.div(ser.sum()) # get ratio
        
    # clean up the order
    v_order = [od for od in index_order if od in ser.index.unique()]
    v_color = pd.Series({k:mpal[k] for k in v_order})
    ser = ser.reindex(v_order)

    if orient == "v":
        bar = ax.bar(x=ser.index, height=ser.values, color=v_color)
        ax.set_xticks(ser.index)
        ax.set_xticklabels(ser.index)
        # adjust the y limits to make it look better
        _, top = ax.get_ylim()
        ax.set_ylim(0, top*1.25)
        
        if label:
            ax.set(ylabel=label)
        else:
            ax.set(ylabel="")

        # remove all spines and axes except the ones with meaningful numbers
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.xaxis.set_visible(False)

    if orient == "h":
        ax.invert_yaxis()
        bar = ax.barh(y=ser.index, width=ser.values, color=v_color)
        ax.set_yticks(ser.index)
        ax.set_yticklabels(ser.index)

        # adjust the x limits to make it look better
        _, right = ax.get_xlim()
        ax.set_xlim(0, right*1.25)

        if label:
            ax.set(xlabel=label)
        else:
            ax.set(xlabel="")
        
        # remove all spines and axes except the ones with meaningful numbers
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.yaxis.set_visible(False)

    return fig, ax, bar


# barplot annotation
def get_pvalue_thresholds(pvalue_format):
    if pvalue_format == "star":
        return ([[1e-4, "****"], [1e-3, "***"],
                    [1e-2, "**"], [0.05, "*"],
                    [1, "ns"]])
    else:
        return( [[1e-5, "1e-5"], [1e-4, "1e-4"],
                   [1e-3, "0.001"], [1e-2, "0.01"],
                   [5e-2, "0.05"]])


def format_p(pvalue, pvalue_format="star", hide_ns=False) -> str:
    """
    Generates simple text for pvalue.
    :param result: test results
    :param pvalue_format: format string for pvalue
    :returns: simple annotation
    """
    # Get and sort thresholds
    pvalue_thresholds = get_pvalue_thresholds(pvalue_format)
    thresholds = sorted(pvalue_thresholds, key=lambda x: x[0])
    #if hide_ns and pvalue_format == "star":
    #    thresholds[-1][1] = ""

    for threshold in thresholds:
        if pvalue <= threshold[0]:
            if pvalue_format == "star":
                pval_text = threshold[1]
            else:
                pval_text = "p ≤ {}".format(threshold[1])
            break
    else: # for ns values
        if hide_ns:
            pval_text = ""
        else:
            pval_text = "p = {}".format('{:.3f}').format(pvalue)

    return pval_text


def annot_barplot_stats(figax, bar, stats_tb, use_stats="Boschloo's padj", pvalue_format="star", hide_ns=False):
    fig, ax = figax # retrieve the barplot axes
    y_ticks = pd.Series({tick.get_text():tick.get_position() for tick in ax.get_yticklabels()}) # retrieve labeling positions # get the test stats
    sig_p = [] # empty list to store returned p values
    # check if stats name is correct
    if use_stats not in stats_tb.columns:
        use_stats = "Boschloo's padj"
    for name in stats_tb[use_stats].index:
        if name in y_ticks.index:
            sig_p.append(format_p(stats_tb[use_stats][name], pvalue_format=pvalue_format, hide_ns=hide_ns))

    ax.bar_label(bar, labels=sig_p, padding=4, color='black', horizontalalignment="left", verticalalignment="bottom")

    return fig, bar
    


def paired_rnalfc_scatter(data_to_plot, masks, titles, orders, colors, simplify=True, lr=False, markersize=14, xlabel=None, ylabel=None, annot_list=None, annot_bool=False, ax_factor=1.5, figax=None):
    
    # retreive rna data columns
    rna_lfc_col1 = [x for x in data_to_plot.columns if re.search("lfc", x) and not re.search("chip", x)][0]
    rna_lfc_col2 = [x for x in data_to_plot.columns if re.search("lfc", x) and not re.search("chip", x)][1]

    if not xlabel:
        xlabel = rna_lfc_col1.replace(".", " ")
    if not ylabel:
        ylabel = rna_lfc_col2.replace(".", " ")
    
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(len(masks), len(masks), sharex=False, sharey=False)
    ax_list = ax_list.flatten()

    if simplify:
        orders = ["Ctrl", "No Diff.", "Diff."]
        colors = ["#CC79A7","#A9A9A9", "#CF8B03"]
        mpal = {a:b for a,b in zip(orders, colors[0:2])}
    else:
        #colors = ["#A9A9A9","#003200","#640000","#000064","#644100"] #dna_classics
        #colors = ["#A9A9A9", "#1A1A1A", "#59A9D8", "#0E927B", "#DC9514"] #dna_safe
        mpal = {a:b for a,b in zip(orders,colors)}
    v_order = [od for od in orders if od in data_to_plot.rna_group.unique()]
    v_color = {k:mpal[k] for k in v_order}

    data_to_plot = data_to_plot.loc[lambda df: df["rna_group"].isin(v_order)]
    
    scatter_paras=dict(
        x = rna_lfc_col1,
        y = rna_lfc_col2,
        xlabel = xlabel,
        ylabel = ylabel,
        hue = "rna_group",
        hue_order = v_order[::-1],
        palette = v_color,
        legend = False,
        lr = True, # always run regression but only add annotation if specified
        s = markersize,
        edgecolor = "none",
        annot_list = annot_list,
        annot_bool = annot_bool,
    )

    lr_text = np.empty(len(masks), dtype=object)

    for i in range(len(masks)):
        fig, ax_list[i], lr_text[i] = scatter_by_category(data=data_to_plot[masks[i]], **scatter_paras, figax=(fig,ax_list[i]))

        # set panel titles
        if titles and len(titles)==len(masks):
            ax_list[i].set_title(titles[i])

    # adjust for axes, make them the same and symmetric, and concentrated a little bit
    xaxis_limits = [abs(x) for sublist in list(zip(*[ax.get_xlim() for ax in ax_list])) for x in sublist]
    yaxis_limits = [abs(x) for sublist in list(zip(*[ax.get_xlim() for ax in ax_list])) for x in sublist]
    axis_limit = max(xaxis_limits + yaxis_limits)
    if ax_factor <= 1.0:
        limit = round(axis_limit + 0.5)
    else:
        limit = round(axis_limit*ax_factor)

    for ax in ax_list:
        # adjust axis limits
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        # add reference lines
        ax.vlines(x=0, ymin=-limit, ymax=limit, linestyle='--', color = 'black', alpha=0.4)
        ax.hlines(y=0, xmin=-limit, xmax=limit, linestyle='--', color = 'black', alpha=0.4)
        # remove y axis labels for subplots to the right
        ss = ax.get_subplotspec()
        if not ss.is_first_col():
            ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
            #ax.set_ylabel("") # high only the title
    
    slope_list = []
    if lr: 
        for text, ax in zip(lr_text, ax_list):
            if scatter_paras["lr"] in ["pearson", "Pearson", "P"]: # add Pearson's R value
                ax.text(limit-0.4, -limit+0.4, text[0], horizontalalignment='right', verticalalignment='bottom', style = 'italic', fontsize=mpl.rcParams["axes.labelsize"])
            elif scatter_paras["lr"] in ["spearman", "Spearman", "S"]:# add Spearman's R value
                ax.text(limit-0.4, -limit+0.4, text[1], horizontalalignment='right', verticalalignment='bottom', style = 'italic', fontsize=mpl.rcParams["axes.labelsize"])
            slope_list.append([text[2], text[3]])

    return fig, ax_list, v_color, slope_list

# pie chart
def pie_by_category(count_ser, label_size=True, add_total=True, sep_wedge=False, labels=None, colors=None, figax=None):
    # set up the pie chart
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplot()
    
    # calculate ratio
    count_ser = count_ser.copy()
    ratio_ser = count_ser.apply(lambda x: x/count_ser.sum())
    # create data
    size = ratio_ser.values
    if label_size: # label percentage
        names = [f"{100*n:.0f}%" for n in ratio_ser.values]
    elif labels:
        names = labels 
    else: # infer label names from pd.Series index
        names = ratio_ser.index

    # Create a circle at the center of the plot
    my_circle = plt.Circle((0,0), 0.5, color='white')

    # Give color names
    if sep_wedge:
        wedges, texts = ax.pie(size, colors=colors, labels=names, wedgeprops = { 'linewidth' : mpl.rcParams["lines.linewidth"], 'edgecolor' : 'white' })
    else:
        wedges, texts = ax.pie(size, colors=colors, labels=names)

    ax.add_artist(my_circle)
    
    if add_total:# add total num at the center of the circle
        ax.text(0,0, str(count_ser.sum().astype(int)), fontsize=mpl.rcParams["axes.titlesize"], va="center", ha="center")

    return fig, ax, wedges, ratio_ser # return wedges for adding legends


# developmental line plot
def single_line_by_category(data, x_column, y_column, xlabel, ylabel, color, figax=None):
    """
    Draw a single line showing the relationship between x and y.

    Parameters
    ----------
    data: pandas DataFrame
        Long format dataframe.

    x_column: str
        Column name of identifier variables to plot on the x-axis.

    y_column: str
        Column name of measured variables to plot on the y-axis.

    xlabel: str
        X-axis label name.

    ylabel: str
        Y-axis label name.

    color: str
        Color of line to be drawn.

    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    
    Returns
    -------
    fig : Figure handle
    ax : Axes handle
    """
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()
    
    ax =  sns.lineplot(data=data, x=x_column, y=y_column, ax=ax, color=color)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    return fig, ax

def dev_line_by_rnaGp(data, morder, mpal, group_col="rna_group", id_col="gene", ages=["p3","p7","p10","p14","p21"], xlabel=None, ylabel=None, uniform_y=False, figax=None):
    """
    Given a categoried gene name identified RNA expression data in wide format, draw a set of line plots for each category specified in the "rna_group" 
    
    Parameters
    ----------
    data: pandas DataFrame
        RNA expression matrix. Row z-score normalized is prefered.

    morder: list
        Order of rna_group category to draw the line plot.
    
    mpal: dictionary
        Dictionary specifying the color of lines.
    
    ages: list
        Column name list of measured varivables to plot.

    xlabel: str
        X-axis label name.

    ylabel: str
        Y-axis label name.

    uniform_y: boolean
        If True, all line plots will use the same y-axis limits.

    figax : (figure, axes) or None
        If specified, make the plot in the provided axes. Otherwise, generate a new axes.
    
    Returns
    -------
    fig : Figure handle
    ax_list: List of axes handles
    """

    data = data.copy()
    
    # retrieve unique groups to plot
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = specseq_plot_utils.setup_multiplot(len(morder), len(morder), sharex=False, sharey=False) # note, do not sharey, it will use the smallest y limits
    ax_list = ax_list.flatten()
    
    # format axes labels:
    if xlabel:
        xlabel = xlabel
    else:
        xlabel = ""
    if ylabel:
        ylabel = ylabel
    else:
        ylabel = ""

    plot_parameters = dict(
        xlabel = xlabel,
        ylabel = ylabel,
        x_column="age",
        y_column="norm.counts"
    )

    for gp, ax in zip(morder, ax_list):
        # reshape data
        data_to_plot = data[data[group_col] == gp].drop(columns=group_col).reset_index(drop=True)
        # convert wide to long dataframe
        data_to_plot = pd.melt(data_to_plot, id_vars=[id_col], value_vars=[age for age in ages if age in data.columns], var_name="age", value_name="norm.counts")
        # order the ages to make sure they match of that specified
        data_to_plot["age"] = data_to_plot["age"].astype(CategoricalDtype(categories=[age for age in ages if age in data.columns], ordered=True))
        fig, ax =  single_line_by_category(data=data_to_plot, color=mpal[gp], **plot_parameters, figax=(fig,ax))
        ax.set(title=gp)

    # adjust for y axes, make them the same
    yaxis_limits = list(zip(*[ax.get_ylim() for ax in ax_list.flat]))
    bottom, top = (min(yaxis_limits[0]), max(yaxis_limits[1]))

    for ax in fig.get_axes():
        ss = ax.get_subplotspec()
        if uniform_y:
            ax.set_ylim(bottom, top)
            if not ss.is_first_col():
                ax.yaxis.set_visible(False) # hide the entire axis, labels, ticks, title
        else:
            if not ss.is_first_col():
                ax.set_ylabel("") # high only the title

    return fig, ax_list
