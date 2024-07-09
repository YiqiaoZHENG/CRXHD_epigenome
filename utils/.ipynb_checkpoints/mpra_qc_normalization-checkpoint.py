import os
import sys
import itertools

# data handling
from scipy import stats
import numpy as np
import pandas as pd

def fraction_by_category(query_BC, ref_df, col_name="recovered"):
    # the query BC should preferentially in pandas Dataframe or pandas Series format
    # otherwise, an error message will arise reminding format conversion

    total_barcodes = ref_df.reset_index(drop=False)[["barcode","annotation","motif"]].groupby(["annotation","motif"]).count().rename(columns={"barcode":"design"})

    # need to add a checking function for columns
    # otherwise need to extract those info from the reference ref_df
    # do it when i have time    
    query_barcodes = query_BC[["annots","annotation","motif"]].groupby(["annotation","motif"]).count().rename(columns={"annots":col_name})

    query_barcodes_byCategory  = pd.merge(total_barcodes, query_barcodes, left_index=True, right_index=True)

    query_barcodes_byCategory[f"fc.{col_name}"] = query_barcodes_byCategory[col_name]/query_barcodes_byCategory.design

    return query_barcodes_byCategory
    

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
