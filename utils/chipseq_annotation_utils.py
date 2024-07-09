import os
import sys
import warnings
import re
import itertools

import numpy as np
import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm

from pybedtools import BedTool

import sklearn
from sklearn.preprocessing import StandardScaler

# intersect two bed-like dataframe
def bedtools_intersect_and_compile(query_df, subject_df, query_prefix="chip", subject_prefix="atac"):
    # the query and subject df should be pre-processed since all peaks will be used for intersection
    query_bed = BedTool.from_dataframe(query_df.copy()) # should i impose requirements for query bed unique name per entry in the fourth column??
    subject_bed = BedTool.from_dataframe(subject_df.copy())

    # intersection
    intersect_bed = query_bed.intersect(subject_bed, loj=True).to_dataframe(header=None).rename(columns={"chrom":"seqnames"})

    # retain only peaks with matched
    intersect_bed = intersect_bed.loc[lambda df: df[intersect_bed.columns[-3]]!=".",:].reset_index(drop=True)

    if query_prefix is not None:
        intersect_bed = intersect_bed.rename(columns={"seqnames":f"{query_prefix}.seqnames", "start":f"{query_prefix}.start", "end":f"{query_prefix}.end"})

    if subject_prefix is not None:
        intersect_bed = intersect_bed.rename(columns={k:f"{subject_prefix}.{v}" for k,v in zip(intersect_bed.columns[-3:], ["seqnames","start","end"])})

    return intersect_bed

# parse homer annotation
def _parse_homer_annotation(full_annotation):
    if type(full_annotation) is str:
        if re.search('\(', full_annotation): # find the parentesis and strip white space
            parsed_annot = re.split(r'(\(+)',full_annotation)[0].strip().lower()
        else:
            parsed_annot = full_annotation.strip().lower()
    else:
        parsed_annot = np.nan
        
    return parsed_annot

def homer_annotation_parser(inTSV_dir, outTSV_dir, annot_columns=["peak.id", "Annotation"], parse_by_column="Annotation", writeToFile=True):
    # read the homer annotation files
    homer_annot = pd.read_csv(inTSV_dir, sep="\t", header=0, low_memory=False)
    # the first column should contain unique peak ID as required by homer
    homer_annot.rename(columns = {list(homer_annot)[0]:'peak.id'}, inplace=True) #rename first column
    # take out only the peak.id and annotation columns
    homer_annot = homer_annot.loc[:, annot_columns].dropna(axis=0, how="any")

    # now parse the annotation column
    homer_annot["annot"] = homer_annot[parse_by_column].apply(_parse_homer_annotation)
    homer_annot = homer_annot.loc[:,annot_columns+["annot"]]
    # sort the column by peak.id
    homer_annot = homer_annot.sort_values(by="peak.id", kind="mergesort").reset_index(drop=True)
    homer_annot["strand"] = "*"

    # write to file if specified
    if writeToFile:
        homer_annot.to_csv(outTSV_dir, sep="\t", index=False, header=True)

    return homer_annot

def get_homer_annot_occr(homer_annt_df, groupby_col="annot", annot_order=None):
    homer_annt_df = homer_annt_df.copy()
    genome_annot = pd.DataFrame(homer_annt_df.groupby(groupby_col).count()["peak.id"]).rename(columns={0:"count"})
    genome_annot["perc"] = genome_annot.apply(lambda row: row / row.sum(), axis=0)
    
    if annot_order is not None:
        # reorder the annotation if specified
        genome_annot = genome_annot.loc[:,annot_order.index].rename(columns=annot_order)
        
    display(genome_annot)
    print(genome_annot.perc.max())
    
    return genome_annot

def grouped_exact_tests(contigency_table, use_test="fisher", alternative="two-sided"):
    # prepare the data
    summary_tb = contigency_table.copy()

    # barnard's
    if use_test == "barnard":
        res = stats.barnard_exact(summary_tb, alternative=alternative)
        return(res.pvalue)

    # fisher's - default
    else:
        res = stats.fisher_exact(summary_tb, alternative=alternative)
        return(res[1])

def odds_ratio_by_Gp(contigency_table, uselog=True):
    # prepare the data
    summary_tb = contigency_table.copy()

    # create a statsmodels Table object
    or_table = sm.stats.Table(summary_tb)
    # retrieve odds ratio
    if uselog:
        odds_ratio = or_table.local_log_oddsratios.iloc[0,0]
    else:
        odds_ratio = or_table.local_oddsratios.iloc[0,0]

    return odds_ratio