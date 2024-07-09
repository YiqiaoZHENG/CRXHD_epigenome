#!/bin/Rscript

print(paste("here i am!!",getwd()))

#process input arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2){
    print("Not enough arguments given. Please give one input bed file and one output bed file full paths.")
}else if (length(args) == 3){
    inputDf <- as.character(args[1])
    outputDf <- as.character(args[2])
    rdsDir <- as.character(args[3])
}else{
    inputDf <- as.character(args[1])
    outputDf <- as.character(args[2])
    print("GREAT analysis will not be saved to RDS. Please provide rds full path directory to save.")
}

#load libraries
suppressPackageStartupMessages(library(rGREAT))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(reshape))


# read bed-like file
cons.peaks <- read.table(file=inputDf, sep="\t", header=TRUE) %>% mutate(strand="*")
cons.peaks <- makeGRangesFromDataFrame(cons.peaks, keep.extra.columns=TRUE) 
# submit for great analysis, smaller the request_interval (in s) faster the processing speed
print(paste("Submitting", basename(inputDf), "for great analysis"))
job <- submitGreatJob(gr=cons.peaks, species="mm10", adv_upstream=5, adv_downstream=1, version="4", request_interval=30)
# retrieve GO enrichment table
#print("Retrieveing GO term enrichment tables")
#gotb <- getEnrichmentTables(job, category = "GO")
# retrieve GRange objects containing the gene-region associations table
print("Retrieving gene-region associations")
# ok....the updated packge is doing something different
# need to check which version of rGREAT is being used on xps/mac/htcf if seeing warning messages
res <- getRegionGeneAssociations(job, request_interval=30)
# flatten the dataframe
parsedRES <- res %>% as_tibble %>% tidyr::unnest(c("annotated_genes","dist_to_TSS"))
write.table(parsedRES, file=outputDf, sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
print(paste("Gene-region association table written to", basename(outputDf)))
# save the job object in rds format
print(job)
if (length(args) == 3){
    saveRDS(c(job, res, parsedRES), file=rdsDir)
    print(paste0("GREAT analysis and results table saved to ",rdsDir))
}

cat("\n")
sessionInfo()
print(paste("any warnings",warnings(),sep=":"))