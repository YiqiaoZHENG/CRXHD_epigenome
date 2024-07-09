#!/bin/Rscript

print("here i am!!")
print(getwd())

suppressPackageStartupMessages(library("ATACseqQC"))
suppressPackageStartupMessages(library("soGGi"))
suppressPackageStartupMessages(library("tidyverse"))

basedir <- getwd() #"/scratch/sclab/100322_ATAC"
outputdir <- file.path(basedir,"DiffBind")
fileoutdir <- file.path(outputdir,"outputs")
figuredir <- file.path(outputdir,"figures")

# Fragment size distribution - use all fragments
# retrieve all available samples
all.samples <- list.files(path = file.path(basedir, "alignedbam"), full.names = F)
# loop over bam files for all samples
getFragDistribution <- function(sample.name){
    # input the ATAC bam file
    bamfile <- file.path(basedir,"alignedbam",sample.name,paste0(sample.name,".sorted.blk.bam"))
    bamfile.labels <- gsub(".sorted.blk.bam", "", basename(bamfile))
    # generate fragement size distribution
    print(paste("Generating fragment distribution file for", bamfile))
    fragSize <- fragSizeDist(bamfile, bamfile.labels)
    # convert the invisible fragment size list to dataframe
    fragSize.df <- as.data.frame(fragSize)
    # write dataframe to text file
    write.table(fragSize.df, file.path(fileoutdir,"fragmentDistribution",paste0(sample.name,".fragDistribution.txt")), sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE)
}
tmp <- sapply(all.samples, function(x) getFragDistribution(x))

cat("\n")
sessionInfo()
print(paste("any warnings",warnings(),sep=":"))
