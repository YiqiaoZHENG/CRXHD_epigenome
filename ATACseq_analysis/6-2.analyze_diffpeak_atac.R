#!/bin/Rscript

print("here i am!!")
print(getwd())

args <- commandArgs(trailingOnly = TRUE)

# tell if running the program in serial or parallel
if (length(args)==2) {
  basedir <- as.character(args[1])
  cores_to_use <- as.numeric(args[2])
  print(paste("running in parallel with ",cores_to_use," cores",sep=""))
  BiocParallel::register(BiocParallel::MulticoreParam(cores_to_use))
}else{
  basedir <- as.character(args[1])
  print("running in serial, if not desired, cancel job and supply a core number")
  BiocParallel::register(BiocParallel::SerialParam()) #default
}

# to check bpparam: 
print(BiocParallel::bpparam())
print(paste("cores detected by parallel",parallel::detectCores(logical = FALSE),sep=":")) #this is what the default use of cores by dba.blacklist()
#print(parallel::detectCores(logical = FALSE))

#load DiffBind libraries
suppressPackageStartupMessages(library(DiffBind))
suppressPackageStartupMessages(library(tidyverse))

#directory variables
#basedir="/scratch/sclab/100322_ATAC"
plotdir=file.path(basedir,"DiffBind","figures")
outputdir=file.path(basedir,"DiffBind","outputs")
beddir=file.path(basedir,"DiffBind","diffbed")

## check metadata spreadsheet
suffix <- "e80a"
data_sheet<-read.csv(file=file.path(basedir, paste0("diffbind_meta.sieved.",suffix,".txt")), sep="\t", header=TRUE)


#read in a set of peaksets and associated metadata
print("here\'s the metadata")
dbObj <- dba(sampleSheet=data_sheet, dir=basedir)
dbObj

# so, let's specify cpu/thread number
if (length(args)==2){
  dbObj$config$cores<-cores_to_use #this is what i asked in the sbatch script
  dbObj$config$RunParallel<-TRUE
}else{
  dbObj$config$RunParallel<FALSE #run in seriels
}


print("generating consensus peaksets")
consObj<-dba.peakset(dbObj, consensus=DBA_FACTOR, minOverlap=2) #overlap replicates
consObj


print("retrieve consensus peakset as GRanges object")
consObj <- dba(consObj, mask=consObj$masks$Consensus, minOverlap=1) #need to specify minOverlap to include all peaks, default is 2
consObj
factor.consensus <- dba.peakset(consObj, bRetrieve=TRUE)
#saveRDS(factor.consensus, file=file.path(outputdir,paste0("100322_atac_peaks.sieved.consensus.",suffix",".rds"))


print("counting with consensus peakset")
#ref for SummarizeOverlaps http://bioconductor.org/packages/release/bioc/vignettes/GenomicAlignments/inst/doc/summarizeOverlaps.pdf
# if summits=TRUE, summits will be calculated and peaksets unaffected
# if summits>0 then all consensus peaks will be re-centered around a consensus summit with width 2*summit+1, default: 200
# score MPRA CRE, peak width and summit as it is
dbObj<-dba.count(dbObj, summits=200, peaks=factor.consensus, bUseSummarizeOverlaps=TRUE) 
dbObj
print("normalizing")
dbObj<-dba.normalize(dbObj,method=DBA_ALL_METHODS,normalize=DBA_NORM_LIB,library=DBA_LIBSIZE_PEAKREADS)
print("contrasting")
dbObj<-dba.contrast(dbObj,categories=DBA_FACTOR,minMembers=2)

print("analyzing")
dbObj<-dba.analyze(dbObj,method=DBA_ALL_METHODS,bRetrieveAnalysis=FALSE)
#note:dba.analyze() retrieves a DBA object with results of analysis added to DBA$contrasts with bRetrieveAnalysis=FALSE
print("analysis done, printing results")
contrast_res<-dba.show(dbObj, bContrasts=T)
print(contrast_res)
analysis_obj_name = paste0("100322_atac_",suffix,"_postanalyze.readsInPeakNormed")
print(paste0("saving post analysis object", analysis_obj_name))
dba.save(dbObj, file=analysis_obj_name, dir=outputdir, pre='dba_', ext='RData', 
         bRemoveAnalysis=FALSE, bRemoveBackground=FALSE,
         bCompress=FALSE)

# retreive peakset post analysis
print("retrieving consesus peakset post analysis")
consensus.peaks <- dba.peakset(dbObj, bRetrieve=TRUE)
seqlevelsStyle(consensus.peaks) <- "UCSC"
saveRDS(consensus.peaks, file=file.path(outputdir,paste0(analysis_obj_name,".rds")))

cat("\n")
sessionInfo()
print(paste0("any warnings:",warnings()))