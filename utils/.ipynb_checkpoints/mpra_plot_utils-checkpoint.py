import os
import sys
import itertools

# plotting functions
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_scatter_density
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager
import matplotlib.gridspec as grid_spec
import seaborn as sns
#import datashader as ds
import plotly
import plotly.express as px
#import kaleido
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
#from joypy import joyplot

# data handling
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity

# aesthetic features - reminder HEX codes are case sensitive

# "Viridis-like" colormap with white background - density plot
white_viridis = LinearSegmentedColormap.from_list("white_viridis", [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

# Okabe & Ito color-blind friendly palette - nested scatter plot
cdict1 = {
    'black': "#000000",
    'orange': "#e69f00",
    'sky blue': "#56b4e9",
    'bluish green': "#009e73",
    'yellow': "#f0e442",
    'blue': "#0072b2",
    'vermilion': "#d55e00",
    'orange': "#e69f00",
    'reddish purple': "#cc79a7",
}

okabe_ito_palette = LinearSegmentedColormap.from_list("okabe_ito", [
    (0, "#000000"), 
    (0.125, "#e69f00"), 
    (0.25, "#56b4e9"),
    (0.375, "#009e73"),
    (0.5, "#f0e442"), 
    (0.625, "#0072b2"), 
    (0.75, "#d55e00"),
    (0.875, "#e69f00"), 
    (1.0, "#cc79a7")
], N=444)

# need to find a better palette to label different CRE or motif category
volcano_fdr_cdict = {"n.s.":'#0e927b', "gain":'#59a9d8', "lost":'#dc9514', "ambiguous":'#979797'}
volcano_annot_cdict = {k:v for k,v in zip(['NotDB','ELost','EGain','KLost', 'KGain','control','RetinalGene'],list(cdict1.values())[:7])}
volcano_motif_cdict = {k:v for k,v in zip(['WT','mutM','mutM','mutDM', 'scrambled'],
                                            ["#000000", "#56B4E9", "#E69F00", "#009E73", "#979797"])}

# color and marker dictionaries for PCA plots by genotype
PCA_cdict = {'wt': "#000000",
            'ehet': "#e69f00",
            'ehom': "#56b4e9",
            'khet': "#009e73",
            'khom': "#f0e442",
            'rhom': "#979797"
}

PCA_mdict_mpl = {'wt': "o",
                'ehet': "v",
                'ehom': ">",
                'khet': "P",
                'khom': "X",
                'rhom': "s"
}

PCA_mdict_plotly = {'wt': "circle",
                    'ehet': "triangle-down",
                    'ehom': "triangle-right",
                    'khet': "cross",
                    'khom': "x",
                    'rhom': "square"
}


# marker styles: https://matplotlib.org/stable/api/markers_api.html
mdict = {
    "circle": "o",
    "triangle down": "v",
    "triangle right": ">",
    "square": "s",
    "cross": "X",
    "plus": "P",
    "diamond": "D",
    "star": "*"
}
# line styles: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

# 1. 2D density plot
def _using_mpl_scatter_density(x, y, label1, label2, figax, 
                               logStretch=True, cmap=white_viridis, 
                               show_stats=False, useLog=True, 
                               draw_reg=False, draw_idt=False, 
                               axis_limits=None):
    
    fig, ax = figax
    
    # Non-linear stretches for high dynamic range data, log strectch density
    if logStretch:
        if logStretch is True:
            logStretch = 400
        norm = ImageNormalize(vmin=0., vmax=logStretch, stretch=LogStretch())
        density = ax.scatter_density(x, y, cmap=cmap, norm=norm)
    else:
        density = ax.scatter_density(x, y, cmap=cmap)
    
    # add regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_lin_reg = np.linspace(min(0, min(x)), max(x), 100)
    y_lin_reg = slope*x_lin_reg+intercept

    # calculate Pearson correlation coefficient
    pearson_corr,_ = stats.pearsonr(x, y)
    # calculate Spearman rank correlation coefficient
    spearman_corr,_ = stats.spearmanr(x, y)

    # format disply text
    text = (f'r = {pearson_corr:.3f}', f'\u03C1 = {spearman_corr:.3f}')

    if useLog:
        # use log scale
        if useLog == 10:
            ax.set_xscale('log', base=useLog)
            ax.set_yscale('log', base=useLog)
        else:
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
        if draw_idt:
            # draw x=y line
            ax.plot(x_lin_reg, x_lin_reg, '--', color = 'grey', alpha=0.8)
        scaleSuffix = "logScale"
    else:
        if draw_reg:
            # draw regression line
            ax.plot(x_lin_reg, y_lin_reg, '--', color = 'orange', alpha=0.8)
        if draw_idt:
            # draw x=y line
            ax.plot(x_lin_reg, x_lin_reg, '--', color = 'grey', alpha=0.8)
        scaleSuffix = "lnrScale"

    if axis_limits is not None:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    if show_stats:
        ax.text(x=.05, y=.95, s=f"$\it r$ = {pearson_corr:.3f}\n$\it \u03C1$ = {spearman_corr:.3f}",
                horizontalalignment='left',
                verticalalignment='top', 
                transform=ax.transAxes)

    ax.set_aspect("equal")

    return fig, ax, density, text

# 2. 2D scatter plot with highlighted data points
def _using_plain_scatter(count_df1, count_df2, label1, label2, 
                         highlight_bc=None, hightlight_label=None, highlight_color="#59A9D8", 
                         show_stats=True, useLog=True, axis_limits=None, draw_idt=True, draw_reg=True,
                         figax=None, figname=None):
    
    # Make a histogram of replicate correlations
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(2,2), dpi=150)

    x=count_df1.copy()
    y=count_df2.copy()

    # add regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_lin_reg = np.linspace(min(0, min(x)), max(x), 100)
    y_lin_reg = slope*x_lin_reg+intercept

    # calculate Pearson correlation coefficient
    pearson_corr,_ = stats.pearsonr(x, y)
    # calculate Spearman rank correlation coefficient
    spearman_corr,_ = stats.spearmanr(x, y)

    # format disply text
    text = (f'r = {pearson_corr:.3f}', f'\u03C1 = {spearman_corr:.3f}', slope, intercept)

    # draw scatter
    ax.scatter(x, y, s=2, color="black", alpha=0.4)
    
    # hight data points if specified
    if highlight_bc is not None:
        ax.scatter(x.loc[highlight_bc], y.loc[highlight_bc], s=2, color=highlight_color, label=hightlight_label)

    if useLog:
        # use log scale
        if useLog == 10:
            ax.set_xscale('log', base=useLog)
            ax.set_yscale('log', base=useLog)
        else:
            ax.set_xscale('log', base=2)
            ax.set_yscale('log', base=2)
        if draw_idt:
            # draw x=y line
            ax.plot(x_lin_reg, x_lin_reg, '--', color = 'grey', alpha=0.8)
        scaleSuffix = "logScale"
    else:
        if draw_reg:
            # draw regression line
            ax.plot(x_lin_reg, y_lin_reg, '--', color = 'orange', alpha=0.8)
        if draw_idt:
            # draw x=y line
            ax.plot(x_lin_reg, x_lin_reg, '--', color = 'grey', alpha=0.8)
        scaleSuffix = "lnrScale"

    if axis_limits is not None:
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    if show_stats:
        ax.text(x=.05, y=.95, s=f"$\it r$ = {pearson_corr:.3f}\n$\it \u03C1$ = {spearman_corr:.3f}", #\n$\it m$ = {slope:.3f}",
                horizontalalignment='left',
                verticalalignment='top', 
                transform=ax.transAxes)

    ax.set_aspect("equal")

    return fig, ax, text


# 3. histogram with defined colors
def _single_hist(data, title, hist_color="#08306B", alpha=0.6, show_mean=False, show_median=False, figax=None, figname=None):
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(2, 2), dpi=150)
    
    data=data.copy()

    ax.hist(data, bins=50, density=True, color=hist_color, alpha=alpha)

    if show_mean:
        ax.axvline(x=data.mean(),
            color="orange",
            ls='--', 
            lw=1)
        ax.text(0.96, 0.92, 'Mean: {:.2f}'.format(data.mean()), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    if show_median:
        ax.axvline(x=data.median(),
            color="green",
            ls='-.', 
            lw=1)
        if show_mean:
            ax.text(0.96, 0.84, 'Median: {:.2f}'.format(data.median()), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)
        else:
            ax.text(0.96, 0.92, 'Median: {:.2f}'.format(data.median()), horizontalalignment='right', verticalalignment='center', transform=ax.transAxes)

    # any other decorations
    if title:
        ax.set_title(title)

    return fig, ax

# 4. ridgeline plot with defined colors, # of samples and genotypes, any additional parameters should be passed using kwargs
def categorical_ridgeline(data, row_by, genotype, # not sure if i want to expand this to multi column plot
                            colors=list(cdict1.values()), alpha=0.6,
                            line_color="k", line_with=0.5,
                            overlap=-0.5, sharex=True, row_order=None,
                            xlabel=None, figax=None, figName=None):

    data = data.copy()
    if row_order is None:
        row_ids = data[row_by].unique()
    else:
        row_ids = row_order
    
    # set up the plot
    if figax is not None:
        fig, ax_array = figax
    else:
        fig = plt.figure(figsize=(2,2), dpi=150)
        gs = (grid_spec.GridSpec(len(row_ids),1))
        # create all the axes first for easy manipulation
        ax_array = []
        [ax_array.append(fig.add_subplot(gs[i:i+1, 0:])) for i in range(len(row_ids))]

    for i,row in enumerate(row_ids):
        ax = ax_array[i]

        # plotting the distribution
        # using pandas.plot function, kde estimate using scipy.stats.gaussian_kde()
        plot = (data[data[row_by] == row].logExp.plot.kde(ax=ax, color=line_color, lw=line_with)) 

        # grabbing x and y data from the kde plot
        x = plot.get_children()[0]._x
        y = plot.get_children()[0]._y

        # filling the space beneath the distribution
        ax.fill_between(x,y,color=colors[i],alpha=alpha)
            
        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax.get_yaxis().set_visible(False)
        #ax.set_xticklabels([])
        # Hide X and Y axes label marks
        #ax.xaxis.set_tick_params(labelbottom=False)
        # Hide X and Y axes tick marks
        #ax.set_xticks([])
        # Hide ticks but keep the labels
        #ax.tick_params(axis='x', which='major', length=0)

        # do slight adjustment based on the axes location
        ss = ax.get_subplotspec()
        if not ss.is_last_row():
            # hide tickss except at bottom plot
            ax.get_xaxis().set_visible(False)
        else:
            # hick tickmarks but not the ticklabels at the bottom plot
            ax.tick_params(axis='x', which='major', length=0)
            ax.set_xlabel("log2 Enhancer activity / Basal")
        
        # add the label to the left of each plot
        ax.text(-0.02, 0, row, fontweight="normal", fontsize=mpl.rcParams["axes.titlesize"],
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes)

        # hide spines expect the bottom one
        [ax.spines[s].set_visible(False) for s in ["top","right","left"]]

    # share x on all axes
    if sharex:
        [ax_array[i].get_shared_x_axes().join(ax_array[i], ax_array[i+1]) for i in range(len(row_ids)-1)]

    # retrieve all x tick positions
    xticks = ax_array[-1].get_xticks()
    for ax in ax_array:
        # adjust the ylims so that the kde looks fatter and shorter
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.4)
        for loc in xticks:
            # add reference lines
            ax.axvline(x=loc, c="grey", lw=0.4, zorder=10)

    # figure title & x axis label
    ax_array[0].set_title(f"Enhancer activity in {genotype} retinas",ha="center",va="bottom")
    if xlabel:
        ax_array[-1].set_xlabel(xlabel)
    else:
        ax_array[-1].set_xlabel("log2 Enhancer activity / Basal")

    # overlap
    gs.update(hspace=overlap)

    if figName is not None:
        figname = figName
    else:
        figname = f"{genotype}_activityBy{row_by}Type"

    return fig, ax_array, figname

# 5. correlation coeff vs variance scatter plot with polynomial fit
def _get_repcoeff_from_cov_cutoff(data, cov_th_list, colidentifier="ratio", cpm_cutoff=0.15): #0.15 cpm corresponding to 3 counts in a 20M library
    # do i want to write this function in a way that it can also take the genotpye information?? 
    # or the input df should pre-processed to include only single genotype information?

    data = data.copy()

    # identity the coeff pairs to be computed and store in an numpy array
    all_reps = data.columns[data.columns.str.contains(colidentifier)]
    rep_pairs = list(itertools.combinations(all_reps,2))
    
    # first initiate a dictionary to store all the camputed numbers
    rep_pearson_dict={key: [] for key in cov_th_list}
    rep_spearman_dict={key: [] for key in cov_th_list}
    rep_numElement_dict={key: [] for key in cov_th_list}

    # then iterate through all cov thresholds to calculate series of coefficient for all rep pairs
    for th in cov_th_list:
        mask = data.loc[:,data.columns[data.columns.str.contains("cov")]].squeeze()<=th
        filtered_data = data.loc[mask, all_reps]
        # total number of BCs passing cov th
        rep_numElement_dict[th].append(len(filtered_data.index))
        # total number of BCs remaining with less than 2 meaningful counts
        rep_numElement_dict[th].append(sum(filtered_data.applymap(lambda count: count>=cpm_cutoff).sum(axis=1)<2))

        for pair in rep_pairs:
            x= np.array(filtered_data.loc[:,pair[0]])
            y= np.array(filtered_data.loc[:,pair[1]])
            # calculate Pearson correlation coefficient
            pearson_corr,_ = stats.pearsonr(x, y)
            # calculate Spearman rank correlation coefficient
            spearman_corr,_ = stats.spearmanr(x, y)
            # store the numbers in the corresponding dictionary
            rep_pearson_dict[th].append(pearson_corr)
            rep_spearman_dict[th].append(spearman_corr)

    # now format the output to dataframe for easy visualization and manipulation
    rep_numElement_df = pd.DataFrame.from_dict(rep_numElement_dict, orient="index", columns=["RemainedBC","LowRepBC"], dtype=np.int32)
    rep_pearson_df = pd.DataFrame.from_dict(rep_pearson_dict, orient="index", columns=[f"{pair[0]}_vs_{pair[1]}" for pair in rep_pairs], dtype=np.float64)
    rep_spearman_df = pd.DataFrame.from_dict(rep_spearman_dict, orient="index", columns=[f"{pair[0]}_vs_{pair[1]}" for pair in rep_pairs], dtype=np.float64)

    return (rep_numElement_df, rep_pearson_df, rep_spearman_df)

def nested_repceoff_vs_cov_scatter(data, cov_th_list, colidentifier="ratio", cpm_cutoff=1, title="RNA samples", line_color_list=cdict1.values(), marker_list=list(mdict.values()), figax=None, figname=None):
    if figax:
        fig, ax_list = figax
    else:
        fig, ax_list = plt.subplots(nrows=2, ncols=2, figsize=(6.8,6.8), dpi=150)
    
    data=data.copy()

    # first i need to generate a list of numbers for plotting
    # call a separate helper function and retreive the output pd.Series
    rep_numElement_df, rep_pearson_df, rep_spearman_df = _get_repcoeff_from_cov_cutoff(data, cov_th_list, colidentifier, cpm_cutoff)

    print(rep_numElement_df)

    # refine the number of markers list
    final_marker_list = marker_list[:len(rep_pearson_df.columns)]

    # plot1 - number of elements remaining
    ax = ax_list[0,0]
    ax.scatter(x=cov_th_list, y=rep_numElement_df.iloc[:,0], color="black", s=4)
    ax.set_ylabel("Num of BCs passing th")
    ax.set_ylim(8000, 18000)

    # plot on the same axes
    #ax2 = ax.twinx()
    #ax2.scatter(x=cov_th_list, y=rep_numElement_df.iloc[:,1], color="saddlebrown", marker="p", s=4)
    #ax2.set_ylabel("Num of BCs with\n<2 meaningful counts")
    #ax2.text(0.04, 0.92, f"cpm cutoff: {str(cpm_cutoff)}", horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
    #ax2.set_ylim(0, ax.get_ylim()[1]*1.1)

    # plot2 - number of elements with at least 2 meaning counts
    ax = ax_list[0,1]
    ax.scatter(x=cov_th_list, y=rep_numElement_df.iloc[:,1], color="saddlebrown", marker="p", s=4)
    ax.set_ylabel("Num of BCs with\n>=2 low counts")
    ax.text(0.04, 0.92, f"cpm cutoff: {str(cpm_cutoff)}", horizontalalignment='left', verticalalignment='center', fontsize=mpl.rcParams["axes.titlesize"], transform=ax.transAxes)
    ax.set_ylim(0, ax.get_ylim()[1]*1.1)

    # plot3 - pearson correlation coefficient
    ax = ax_list[1,0]
    rep_pearson_df = rep_pearson_df.reset_index(drop=False).rename(columns={"index":"cov_th"})
    melted_pearson = pd.melt(rep_pearson_df, id_vars=["cov_th"], value_vars=rep_pearson_df.columns[1:], var_name="cmpar", value_name="corr")
    ax = sns.scatterplot(data=melted_pearson, x="cov_th", y="corr", hue="cmpar", style="cmpar", palette=line_color_list, markers=final_marker_list, s=8, ax=ax, legend=False)
    # now move the legend out of the drawing space
    #handles, labels  =  ax.get_legend_handles_labels()
    #ax.legend(handles, labels, loc=(1.02, 0))
    ax.set_ylabel(f"Pearson's $\it r$")

    # plot4 - spearson correlation coefficient 
    ax = ax_list[1,1]
    rep_spearman_df = rep_spearman_df.reset_index(drop=False).rename(columns={"index":"cov_th"})
    melted_spearman = pd.melt(rep_spearman_df, id_vars=["cov_th"], value_vars=rep_pearson_df.columns[1:], var_name="cmpar", value_name="corr")
    ax = sns.scatterplot(data=melted_pearson, x="cov_th", y="corr", hue="cmpar", style="cmpar", palette=line_color_list, markers=final_marker_list, s=8, ax=ax)
    # now move the legend out of the drawing space
    handles, labels  =  ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=(1.1, 0))
    ax.set_ylabel(f"Spearman's $\it \u03C1$")

    small_ylim = min(melted_pearson["corr"].min(), melted_spearman["corr"].min())
    [ax.set_ylim(small_ylim-0.02, 1.0) for ax in [ax_list[1,0], ax_list[1,1]]]


    [ax.set_xlabel("cov threshold") for ax in ax_list.flatten()]
    [ax.set_xlim(np.array(cov_th_list).min()-0.2,np.array(cov_th_list).max()+0.2) for ax in ax_list.flatten()]

    fig.suptitle(title)
    fig.tight_layout()

    figname = f"{figname}.CovarianceThreholdScatter"

    return fig, ax_list, figname

# 6. PCA plot with all samples
def _parse_sampletype(sampleName):
    if "rna" in sampleName:
        return "rna"
    elif "dna" in sampleName:
        return "dna"
    elif "plasmid" in sampleName:
        return "plasmid"

def _parse_genotype(sampleName):
    if "rna" in sampleName:
        return sampleName[4:]
    elif "dna" in sampleName:
        return sampleName[3:]
    elif "plasmid" in sampleName:
        return "plasmid"

def PCA_all_genotypes(data_for_PCA, colmask, colorby="genotype", PCA_cdict=PCA_cdict, PCA_mdict=PCA_mdict_plotly, pca_dim=3):
    # transpose data so that samples by rows and BCs by columns
    if colmask is not None:
        data_for_PCA = data_for_PCA.loc[:,data_for_PCA.columns[colmask]].copy().T
    else:
        data_for_PCA = data_for_PCA.copy().T
    
    # define the features to fit PCA
    features = data_for_PCA.columns

    # categorization for aesthetic purpose
    data_for_PCA = data_for_PCA.reset_index(drop=False).rename(columns={"index":"sampleName"})
    # annotate sampletype and genotype
    data_for_PCA["sampletype"] = data_for_PCA["sampleName"].apply(lambda x: _parse_sampletype(x))
    data_for_PCA["genotype"] = data_for_PCA["sampleName"].apply(lambda x: _parse_genotype(x))

    # make the cdict and mdict has the appropraitedly assigned values for all the compenents to be plotted

    # dimensional reduction and plot
    pca = PCA()
    components = pca.fit_transform(data_for_PCA[features])
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(pca_dim),
        color=data_for_PCA[colorby],
        symbol=data_for_PCA[colorby],
        color_discrete_sequence = list(PCA_cdict.values()),
        symbol_sequence=list(PCA_mdict_plotly.values()), #plotly marker need to be specified as on their website https://plotly.com/python/marker-style/
        width=640, height=600,
    )

    # hide diagonal plots
    fig.update_traces(diagonal_visible=False)

    return fig

#access rep corr via scatter plot across different cov threhold
def cov_th_rep_corr_scatter(data, cov_th_list, annotation, colmask=None, 
                            highlight_bc=None, hightlight_label=None, highlight_color="#59A9D8", 
                            show_stats=True, useLog=True, axis_limits=None, draw_reg=True,
                            figtitle=None, figax=None, figName=None):
    if figax:
        fig, ax_list = figax
    else:
        colnum = len(colmask)
        rownum = len(cov_th_list)
        fig, ax_list = plt.subplots(nrows=rownum, ncols=colnum*colnum, sharex=True, sharey=True, figsize=(2*colnum,2*rownum), dpi=150)

    data = data.copy()

    # identity the sample pairs to be computed and store in a numpy array
    if colmask is not None:
        all_reps=colmask
    else:
        all_reps=data.columns
    rep_pairs = list(itertools.combinations(all_reps,2))

    # always include basal in plots
    basal_mask = data.index.isin(annotation.index[annotation["label"].str.contains("basal")])

    for i,th in enumerate(cov_th_list):
        mask = data.loc[:,data.columns[data.columns.str.contains("cov")]].squeeze()<=th
        filtered_data = data.loc[basal_mask+mask, :]

        ax_list[i,0].text(x=.5, y=.5, s=f"cov th: {str(th)}", horizontalalignment='center', verticalalignment='center', fontsize=mpl.rcParams["axes.titlesize"], transform=ax_list[i,0].transAxes)
        ax_list[i,0].axis("off")

        for j,pair in enumerate(rep_pairs):
            fig, ax_list[i,j+1], _ = _using_plain_scatter(filtered_data[pair[0]], filtered_data[pair[1]], pair[0], pair[1], 
                                                        highlight_bc=highlight_bc, hightlight_label=hightlight_label, highlight_color=highlight_color,
                                                        show_stats=show_stats, useLog=useLog, draw_reg=draw_reg,
                                                        axis_limits=axis_limits, figax=(fig,ax_list[i,j+1]))

    fig.tight_layout()

    if figName:
        figname = f"{figName}_repcorrScatter"
    else:
        figname = "repcorrScatter"

    return fig, ax_list, figname
    
def cov_th_rep_corr_density(data, cov_th_list, annotation, figsize=(7.5,9), colmask=None, 
                            logStretch=True, cmap=white_viridis, 
                            show_stats=False, useLog=True, 
                            draw_reg=False, draw_idt=False, 
                            axis_limits=None,
                            figtitle=None, figName=None):
    data = data.copy()

    # always include basal in plots
    basal_mask = data.index.isin(annotation.index[annotation["label"].str.contains("basal")])
    
    if colmask is not None:
        all_reps=colmask
    else:
        all_reps=data.columns[:-3] # assume only mean, std and cov are at the end of the dataframe, using mask is advised

    # identity the sample pairs to be computed and store in a numpy array
    rep_pairs = list(itertools.combinations(all_reps,2))

    colnum = len(rep_pairs)+1
    rownum = len(cov_th_list)

    # set up the figure
    fig = plt.figure(figsize=figsize, dpi=150, constrained_layout=True)

    gs = fig.add_gridspec(nrows=rownum, ncols=colnum, width_ratios=[1]*colnum, height_ratios=[1]*rownum)
    gs.update(left=.05,right=.95,top=.95,bottom=.05) #padding for each grid space

    # setup all axes first for easy manipulation
    ax_array = []
    for i in range(rownum):
        ax_array.append(fig.add_subplot(gs[i, 0]))
        for j in range(colnum-1):
            ax_array.append(fig.add_subplot(gs[i, j+1], projection='scatter_density'))
    ax_array = np.array(ax_array).reshape(rownum, colnum)

    for i,th in enumerate(cov_th_list):
        mask = data.loc[:,data.columns[data.columns.str.contains("cov")]].squeeze()<=th
        filtered_data = data.loc[basal_mask+mask, :]

        # first add the cov th label
        ax = ax_array[i,0]
        ax.text(x=.5, y=.5, s=f"cov th: {str(th)}", horizontalalignment='center', verticalalignment='center', fontsize=mpl.rcParams["axes.titlesize"], transform=ax.transAxes)
        ax.axis("off")

        for j,pair in enumerate(rep_pairs):
            ax = ax_array[i,j+1]
            fig, ax, density, _ =  _using_mpl_scatter_density(filtered_data[pair[0]], filtered_data[pair[1]], pair[0], pair[1],
                                                                cmap=cmap, logStretch=logStretch,
                                                                show_stats=show_stats, useLog=useLog,
                                                                draw_reg=draw_reg, draw_idt=draw_idt,
                                                                axis_limits=axis_limits,
                                                                figax=(fig,ax))
            
    # hide the aixs labels if not at the bottom row similar to sharex sharey in plt.subplots


    if figName:
        figname = f"{figName}_repcorrDensity"
    else:
        figname = "repcorrDensity"

    return fig, ax_array, figname

def categorical_rep_corr_scatter(data, category_list, annotation, colmask=None, cov_th=1.0,
                                    highlight_bc=None, hightlight_label=None, highlight_color="#59A9D8", 
                                    show_stats=True, useLog=True, axis_limits=None, draw_reg=True,
                                    figtitle=None, figax=None, figName=None):
    
    data = data.copy()

    # always include basal in plots
    basal_index = annotation.index[annotation["label"].str.contains("basal")]
    basal_data = data.loc[list(set(data.index) & set(basal_index))]
    

    if colmask is not None:
        all_reps=colmask
    else:
        all_reps=data.columns[:-3] # assume only mean, std and cov are at the end of the dataframe, using mask is advised

    # identity the sample pairs to be computed and store in a numpy array
    rep_pairs = list(itertools.combinations(all_reps,2))

    if figax:
        fig, ax_list = figax
    else:
        colnum = len(rep_pairs)+1
        rownum = len(category_list)
        fig, ax_list = plt.subplots(nrows=rownum, ncols=colnum, sharex=True, sharey=True, figsize=(1.2*colnum,1.2*rownum), dpi=150)
    # add a small function to check whether all category specified exist in motif and annotation category

    print(f"filtering BCs by cov th: {str(cov_th)}")

    for i,category in enumerate(category_list):
        # filter by cov threshold
        filtered_index = data.index[data.loc[:,data.columns[data.columns.str.contains("cov")]].squeeze()<=cov_th]
        # overlapping BCs in the specified category and the input data
        filtered_data = data.loc[list(set(annotation.index[annotation.motif==category]) & set(filtered_index))]
        # attach basal data
        filtered_data = pd.concat([filtered_data, basal_data], axis=0, join="outer", ignore_index=False)

        # first add the CRE type label
        ax = ax_list[i,0]
        ax.text(x=.5, y=.5, s=f"motif type: {category}\nn = {str(len(filtered_data.index))}\ncov th = {str(cov_th)}",
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=mpl.rcParams["axes.titlesize"], transform=ax.transAxes)
        ax.axis("off")

        for j,pair in enumerate(rep_pairs):
            fig, ax_list[i,j+1], _ = _using_plain_scatter(filtered_data[pair[0]], filtered_data[pair[1]], pair[0], pair[1], 
                                                        highlight_bc=highlight_bc, hightlight_label=hightlight_label, highlight_color=highlight_color,
                                                        show_stats=show_stats, useLog=useLog, draw_reg=draw_reg,
                                                        axis_limits=axis_limits, figax=(fig,ax_list[i,j+1]))

    # hide the aixs labels if not at the bottom row similar to sharex sharey in plt.subplots

    
    fig.tight_layout()
    

    if figName:
        figname = f"{figName}_repcorrDensity"
    else:
        figname = "repcorrDensity"

    return fig, ax_list, figname

def categorical_rep_corr_density(data, category_list, annotation, cov_th=1.0, figsize=(7.5,9), colmask=None,
                                 cmap=white_viridis, logStretch=True,
                                 show_stats=True, useLog=2,
                                 draw_reg=False, draw_idt=True,
                                 axis_limits=None,
                                 figtitle=None, figName=None):
    data = data.copy()

    print(f"filtering BCs by cov th: {str(cov_th)}")
    # always include basal in plots
    basal_index = annotation.index[annotation["label"].str.contains("basal")]
    basal_data = data.loc[list(set(data.index) & set(basal_index))]
    
    if colmask is not None:
        all_reps=colmask
    else:
        all_reps=data.columns[:-3] # assume only mean, std and cov are at the end of the dataframe, using mask is advised

    # identity the sample pairs to be computed and store in a numpy array
    rep_pairs = list(itertools.combinations(all_reps,2))

    # add a small function to check whether all category specified exist in motif and annotation category
     
    colnum = len(rep_pairs)+1
    rownum = len(category_list)

    # set up the figure
    fig = plt.figure(figsize=figsize, dpi=150, constrained_layout=True)

    gs = fig.add_gridspec(nrows=rownum, ncols=colnum, width_ratios=[1]*colnum, height_ratios=[1]*rownum)
    gs.update(left=.05,right=.95,top=.95,bottom=.05) #padding for each grid space


    # setup all axes first for easy manipulation
    ax_array = []
    for i in range(rownum):
        ax_array.append(fig.add_subplot(gs[i, 0]))
        for j in range(colnum-1):
            ax_array.append(fig.add_subplot(gs[i, j+1], projection='scatter_density'))
    ax_array = np.array(ax_array).reshape(rownum, colnum)

    for i,category in enumerate(category_list):
        # filter by cov threshold
        filtered_index = data.index[data.loc[:,data.columns[data.columns.str.contains("cov")]].squeeze()<=cov_th]
        # overlapping BCs in the specified category and the input data
        filtered_data = data.loc[list(set(annotation.index[annotation.motif==category]) & set(filtered_index))]
        # attach basal data
        filtered_data = pd.concat([filtered_data, basal_data], axis=0, join="outer", ignore_index=False)

        # first add the CRE type label
        ax = ax_array[i,0]
        ax.text(x=.5, y=.5, s=f"motif type: {category}\nn = {str(len(filtered_data.index))}\ncov th = {str(cov_th)}",
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=mpl.rcParams["axes.titlesize"], transform=ax.transAxes)
        ax.axis("off")

        for j,pair in enumerate(rep_pairs):
            ax = ax_array[i,j+1]
            fig, ax, density, _ =  _using_mpl_scatter_density(filtered_data[pair[0]], filtered_data[pair[1]], pair[0], pair[1], 
                                                                cmap=cmap, logStretch=logStretch,
                                                                show_stats=show_stats, useLog=useLog,
                                                                draw_reg=draw_reg, draw_idt=draw_idt,
                                                                axis_limits=axis_limits,
                                                                figax=(fig,ax))
            
    # hide the aixs labels if not at the bottom row similar to sharex sharey in plt.subplots
    

    if figName:
        figname = f"{figName}_repcorrDensity"
    else:
        figname = "repcorrDensity"

    return fig, ax_array, figname

def _get_lognormal_params(row):
    """Helper function to get parameters of lognormal distribution from linear data.

    Parameters
    ----------
    row : pd.Series
        Row of a df corresponding to barcode averages in each replicate.

    Returns
    -------
    params : pd.Series
        mu and sigma for the lognormal distribution, and the number of replicates the sequence was measured in.
    """
    mean = row.mean()
    std = row.std()
    cov = std / mean

    # Rely on the fact that the mean is exp(mu + 1/2 sigma**2) and the variance is mean**2 * (exp(sigma**2) - 1)
    log_mean = np.log(mean / np.sqrt(cov**2 + 1))
    log_std = np.sqrt(np.log(cov**2 + 1))
    params = pd.Series({
        "mean": log_mean,
        "std": log_std,
        "n": row.count()
    })

    return params

def log_ttest_vs_basal(df, basal_key):
    """Do t-tests in log space to see if sequences has the same activity as basal.

    Parameters
    ----------
    df : pd.DataFrame
        Index is sequence ID, columns are average RNA/DNA barcode counts for each replicate.
    basal_key : str
        Index value for basal.

    Returns
    -------
    pvals : pd.Series
        p-value for t-test of the null hypothesis that the log activity of a sequence is the same as that of basal.
        Does not include a p-value for basal.
    """
    log_params = df.apply(_get_lognormal_params, axis=1)

    # Pull out basal params
    basal_mean, basal_std, basal_n = log_params.loc[basal_key]

    # Drop basal from the df
    log_params = log_params.drop(index=basal_key)

    # Do t-tests on each row
    pvals = log_params.apply(lambda x: stats.ttest_ind_from_stats(basal_mean, basal_std, basal_n,
                                                                  x["mean"], x["std"], x["n"],
                                                                  equal_var=False)[1],
                             axis=1)
    return pvals

def fdr(pvalues, name_prefix=None):
    """Correct for multiple hypotheses using Benjamini-Hochberg FDR and return q-values for each observation. Ties
    are assigned the largest possible rank.

    Parameters
    ----------
    pvalues : pd.Series or pd.DataFrame
        Each row is the p-value for an observation. If pvalues is a DataFrame, each column is a different condition.
        FDR is performed separately on each column.
    name_prefix : str, list, or None
        Prefix(es) to use for name(s) of the q-values. `_qvalue` is appended to the prefixes If a str, then pvalues
        must be a Series; if list-like, then pvalue must be a DataFrame. If None or a datatype mismatch, simply take
        the old names and append `_qvalue` to the names.

    Returns
    -------
    qvalues : Same as pvalues
        The FDR-corrected q-values.

    """
    n_measured = pvalues.notna().sum()
    ranks = pvalues.rank(method="max")
    qvalues = pvalues * n_measured / ranks
    suffix = "_qvalue"

    # Define the name of qvalues
    if type(pvalues) is pd.Series:
        if type(name_prefix) is str:
            name_prefix += suffix
        else:
            name_prefix = pvalues.name + suffix
        qvalues.name = name_prefix

    elif type(pvalues) is pd.DataFrame:
        if type(name_prefix) is not list and type(name_prefix) is not np.array:
            name_prefix = pvalues.columns

        name_prefix = [i + suffix for i in name_prefix]
        qvalues.columns = name_prefix
    else:
        raise Exception(f"Error, pvalues is not a valid data type (this should never happen), it is a {type(pvalues)}")
    return qvalues

def _volcano_color_mapper(data, by="fdr", cdict=volcano_fdr_cdict, lfc_th=1, fdr_th=0.05):
    data = data.copy()
    if "logExp" not in data.columns:
        data["logExp"] = data["expression"].apply(np.log2)
    # add a small function to check if the given color dict has enough colors for the catergization
    if by == "fdr":
        # annotate by log2fc and fdr
        data["color"] = cdict["ambiguous"]
        data[["color"]] = data[["color"]].mask((data.logExp >= lfc_th) & (data["expression_qvalue"] < fdr_th), cdict["gain"], inplace=False)
        data[["color"]] = data[["color"]].mask((data.logExp <= -lfc_th) & (data["expression_qvalue"] < fdr_th), cdict["lost"], inplace=False)
        data[["color"]] = data[["color"]].mask((abs(data.logExp) < lfc_th) & (data["expression_qvalue"] >= fdr_th), cdict["n.s."], inplace=False)
    elif by == "annotation":
        # annotate by CRE categorization
        data["color"] = data["annotation"].apply(lambda type: cdict[type])
    elif by == "motif":
        # annotate by motif type
        data["color"] = data["motif"].apply(lambda type: cdict[type])
    elif by in data.columns:
        # annotate by customized parameter
        data["color"] = data[by].apply(lambda type: cdict[type])
    else:
        print(f"The given {by} is not found in data columns. Not color map generated.")
        
    return data

# make this into a stand-alone function with color map and alpha utilities
def volcano_with_colorCode(data, color_by="fdr", color_dict=volcano_fdr_cdict, 
                            add_ref=True, xref=1, yref=0.05, x_useLog=True,
                            xlabel=None, ylabel=None, title=None, legend_title=None,
                            figax=None, figName=None):

    data = data.copy()

    # annotate scatter dot colors
    data = _volcano_color_mapper(data, by=color_by, cdict=color_dict, lfc_th=xref, fdr_th=yref)

    # set up the plot
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(4,2), dpi=150)
    
    if x_useLog:
        ax.scatter(x=data["expression"].apply(lambda x:np.log2(x)),
                    y=data["expression_qvalue"].apply(lambda x:-np.log10(x)),
                    s=1, c=data["color"])
    else:
        ax.scatter(x=data["expression"],
                    y=data["expression_qvalue"].apply(lambda x:-np.log10(x)),
                    s=1, c=data["color"])

    if color_by == "fdr": 
        summary_nums = [sum(data.color == color_dict[type]) for type in ["lost","gain","n.s."]]
        summary_nums.append(len(data.index))

        # add the text of each category
        ax.text(0.01,0.99,s=f"n={str(summary_nums[0])}", fontsize=mpl.rcParams["axes.labelsize"], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.99,0.99,s=f"n={str(summary_nums[1])}", fontsize=mpl.rcParams["axes.labelsize"], horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.01,0.01,s=f"n={str(summary_nums[2])}", fontsize=mpl.rcParams["axes.labelsize"], horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
        ax.text(0.99,0.01,s=f"total={str(summary_nums[3])}", fontsize=mpl.rcParams["axes.labelsize"], horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    # add reference lines
    if add_ref:
        ax.axvline(-abs(xref),color="k",linestyle="--")
        ax.axvline(abs(xref),color="k",linestyle="--")
        ax.axhline(-np.log10(yref),color="k",linestyle="--")

    # aesthetics
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("log2 Enhancer activity / Basal")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("-log10FDR")
    if title is not None:
        ax.set_title(title)

    # customized legend
    if legend_title is None:
        legend_title = ""
    legend_elements = [mpl.lines.Line2D([0], [0], marker='o', color=color, label=label, lw=0, 
                                        markersize=3) for label,color in color_dict.items()]
    ax.legend(handles=legend_elements, loc=(1.02,0), frameon=False, title=legend_title)

    if figName:
        figname = figName
    else:
        figname = f"activityFDR.colorBy{str.capitalize(color_by)}"

    return fig, ax, figname

