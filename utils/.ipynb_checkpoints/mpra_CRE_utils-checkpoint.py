import os
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from utils import sequence_annotator

# I. Find matched motifs and do something
# find the matching motif using the unparsed fimo
def find_motif_match(fimo_df, motif_name, coremotif_dict=None):
    # core motif dictionary are 1-indexed, matches specseq ewm convention
    fimo_df = fimo_df.copy()

    if motif_name not in fimo_df.motif_id.unique():
        raise ValueError(f"Motif {motif_name} not in fimo dataframe.")
    else:
        small_fimo_score = fimo_df.loc[lambda df: df.motif_id == motif_name,:].copy().reset_index(drop=True)

        # find perfect match with the core motif
        if coremotif_dict:
            splited_sequences = small_fimo_score.matched_sequence.str.split("", expand = True)
            small_fimo_score["match"] = (splited_sequences[list(coremotif_dict.keys())] == list(coremotif_dict.values())).all(1)
        else:
            small_fimo_score["match"] = True

        match_fimo_score = small_fimo_score.loc[lambda df: df.match==True,:].reset_index(drop=True)

        return small_fimo_score, match_fimo_score


# locate and mask matching motif instances
def mask_matched_motif(fasta_ser, matched_fimo_df):
    masked_fasta = {}

    matched_peaks = matched_fimo_df["sequence_name"].unique().tolist()

    for identifier, seq in fasta_ser.iteritems():
        if identifier in matched_peaks:
            # find the fimo result entry if exists
            fimo_matched = matched_fimo_df[matched_fimo_df["sequence_name"] == identifier].reset_index(drop=True)
            # iterate through all fimo matched entries
            for match in fimo_matched.index:
                start = int(fimo_matched.at[match, "start"])
                end = int(fimo_matched.at[match, "stop"])
                match_seq = fimo_matched.at[match, "matched_sequence"]

                # mask the matched substring into strings of N in the fasta record
                masked_seq = seq.replace(seq[start-1:end], "N"*len(match_seq))
                # update the fasta sequence
                seq = masked_seq

        # update the fasta and match count dictionary
        masked_fasta[identifier] = seq

    # convert dictionary to Series
    masked_fasta = pd.Series(masked_fasta)
    masked_fasta.index.name = "label"

    return masked_fasta

# wrapper function to mask matched motifs
def find_and_mask_motif(fasta_ser, raw_fimo_df, motif_name, coremotif_dict=None):
    fasta_ser = fasta_ser.copy()
    raw_fimo_df = raw_fimo_df.copy()

    # identify all motif intances that are perfect matches at core positions
    _, match_fimo_score = find_motif_match(raw_fimo_df, motif_name, coremotif_dict=coremotif_dict)

    # mutate all perfect matched motif instances
    masked_fasta = mask_matched_motif(fasta_ser, match_fimo_score)

    return match_fimo_score, masked_fasta


def mutate_motif_bystrand(string, strand, mutate_dict):
    # Replace characters at index positions in list
    # Note the motif dictionary is 1-indexed
    for i,j in mutate_dict.items():
        string = string[:i-1] + j + string[i:]
    # reverse the sequence before attaching it back to the fasta
    if strand == "-":
        string = sequence_annotator.rev_comp(string)
    
    return string

# locate and mutate matching motif instances
def mutate_matched_motif(fasta_ser, matched_fimo_df, motif_name, mutantmotif_dict):
    # mutant motif dictionary are 1-indexed, matches specseq ewm convention
    mutated_fasta = {}
    match_fimo_score = []

    matched_peaks = matched_fimo_df["sequence_name"].unique().tolist()

    for identifier, seq in fasta_ser.iteritems():
        if identifier in matched_peaks:
            # find the fimo result entry if exists
            fimo_matched = matched_fimo_df[matched_fimo_df["sequence_name"] == identifier].reset_index(drop=True)
            # iterate through all fimo matched entries
            for match in fimo_matched.index:
                start = int(fimo_matched.at[match, "start"])
                end = int(fimo_matched.at[match, "stop"])
                strand = fimo_matched.at[match, "strand"]
                score = float(fimo_matched.at[match, "score"])
                match_seq = fimo_matched.at[match, "matched_sequence"]
    
                # mutant the matched substring with positions and nucleotides specified
                mutated_seq = mutate_motif_bystrand(match_seq, strand, mutantmotif_dict)
                match_fimo_score.append([identifier, motif_name, start, end, strand, score, match_seq, mutated_seq])
                # replace the matched motif with the mutated version
                mutated_seq = seq.replace(seq[start-1:end], mutated_seq)
                # update the fasta sequence
                seq = mutated_seq

        # update the fasta and match record dictionary
        mutated_fasta[identifier] = seq
        
    # convert dictionary to Series
    mutated_fasta = pd.Series(mutated_fasta)
    mutated_fasta.index.name = "label"

    match_fimo_score = pd.DataFrame(match_fimo_score, columns = ['peak.id', 'motif', "start", "end", "strand", "score", "match_seq", "mutated_seq"])

    return match_fimo_score, mutated_fasta

# wrapper function to mutate matched motifs
def find_and_mutate_motif(fasta_ser, raw_fimo_df, motif_name, mutantmotif_dict, coremotif_dict=None):
    fasta_ser = fasta_ser.copy()
    raw_fimo_df = raw_fimo_df.copy()

    # identify all motif intances that are perfect matches at core positions
    _, match_fimo_score = find_motif_match(raw_fimo_df, motif_name, coremotif_dict=coremotif_dict)

    # mutate all perfect matched motif instances
    match_fimo_score, mutated_fasta = mutate_matched_motif(fasta_ser, match_fimo_score, motif_name, mutantmotif_dict)

    return match_fimo_score, mutated_fasta


# find sequences containing restriction enzyme cut site sequences
def find_REsite_match(fasta_ser, RE_list):
    fasta_ser = fasta_ser.copy()
    RE_list = "|".join(RE_list)
    print(f"Looking for matches: {RE_list}")
    RE_matched = fasta_ser[fasta_ser.str.contains(pat = RE_list)]

    return RE_matched


# count the occurrence of a specified motif
def count_motif_occur(fasta_ser, fimo_df, motif_name, coremotif_dict=None):
    fimo_df = fimo_df.copy()
    match_count = pd.DataFrame(index=fasta_ser.index.copy())

    _, match_fimo_score = find_motif_match(fimo_df=fimo_df, motif_name=motif_name, coremotif_dict=coremotif_dict)
    
    # palindrome motif only counted as one
    count_df = match_fimo_score.groupby(["sequence_name", "start"]).size().reset_index()
    count_df = count_df.groupby(["sequence_name"]).size().reset_index().rename(columns={0:"motif_count", "sequence_name":"peak.id"}).set_index("peak.id")

    match_count = pd.merge(match_count, count_df, left_index=True, right_index=True, how="outer").fillna(0)

    return match_count


# calculate GC content for all instances in a fasta series
def calculate_fasta_GC(fasta_ser):
    fasta_ser = fasta_ser.copy()
    # expand all fasta sequnces to individual characters
    fasta_ser = fasta_ser.str.split("", expand = True)
    # get length of individual CRE
    allCRE_GC = fasta_ser.count(axis=1).to_frame(name="CRElen")-2 # account for the start and ending characters
    # get total GC count for individual CRE
    allCRE_GC["GCcount"] = fasta_ser.applymap(lambda x: (x=="G")|(x=="C"), na_action='ignore').sum(axis=1)
    # calculate GC percentage
    allCRE_GC["GCperc"] = allCRE_GC["GCcount"]/allCRE_GC["CRElen"]

    return allCRE_GC


# calculate position specific GC content for all instances in a fasta seires
# note all input fasta instances need to be of the same length
def calculate_position_GC(fasta_ser):
    fasta_ser = fasta_ser.copy()
    # expand all fasta sequnces to individual characters
    fasta_ser = fasta_ser.str.split("", expand = True)
    # get length of individual CRE
    allCRE_GC = fasta_ser.count(axis=1).to_frame(name="CRElen")
    # check if all CREs are of the same length
    if len(allCRE_GC.CRElen.unique()) > 1:
        raise ValueError("Size of CREs are not uniform.")
    else:
        # get total number of CREs
        allCRE_GC = fasta_ser.drop(columns=[0,len(fasta_ser.columns)-1]).count(axis=0).to_frame(name="CRElen")
        # get total GC count for individual CRE
        allCRE_GC["GCcount"] = fasta_ser.applymap(lambda x: (x=="G")|(x=="C"), na_action='ignore').sum(axis=0)
        # calculate GC percentage
        allCRE_GC["GCperc"] = allCRE_GC["GCcount"]/allCRE_GC["CRElen"]

        return allCRE_GC


# III.

def make_N_CREs(CRE_df, N):
    CRE_df = CRE_df.copy()

    reps = [int(N)]*len(CRE_df.index)
    replicated_CRE_df = CRE_df.loc[np.repeat(CRE_df.index.values, reps)]

    return replicated_CRE_df
    