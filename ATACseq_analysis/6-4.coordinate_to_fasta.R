#!/bin/Rscript

# check number and dtype of arguments passed to script
cmd_args <- commandArgs(TRUE)
if (length(cmd_args) < 3) stop("Not enough arguments. Please supply 3 arguments.")

inputBed <-  as.character(cmd_args[1])
indexColumn <- as.character(cmd_args[2])
outputFasta <-  as.character(cmd_args[3])

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(BSgenome.Mmusculus.UCSC.mm10))

genomedb <- BSgenome.Mmusculus.UCSC.mm10

print(paste("Reading coordinates from",inputBed))
print(paste("Using column",indexColumn,"as fasta names"))

# read in bed-like file
# the input bed file should have an uniuqe identifier for each genomic coordinate as index column
df <- read.table(inputBed, sep="\t", header = TRUE) %>% column_to_rownames(var=indexColumn)
# convert to GRanges object
gr <- makeGRangesFromDataFrame(df, keep.extra.columns = TRUE) %>% sort()

# retreive fasta sequences
seq <- getSeq(genomedb, gr)

print(paste("Writing fasta sequences to",outputFasta))
writeXStringSet(seq , filepath=outputFasta, 
                    append=FALSE, compress=FALSE, compression_level=NA, format="fasta")

cat("ha! this is the end of the coordinate_to_fasta.R script!\n")

# print warnings is any
if (!is.null(warnings())) {message(paste("any warnings",warnings(),sep = ":"))}
