#!/bin/bash

#SBATCH --job-name=yq_great
#SBATCH --nodes=1
# --ntasks-per-node=8
#SBATCH --mem=32G
# --cpus-per-task=8
# --mem-per-cpu=16G
# --exclusive
#SBATCH --time=02:00:00
# --array=1-5
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_great/great_%j.out

echo "Number of CPUS on the allocated node ${SLURM_CPUS_ON_NODE}"
echo "Count of processors available to the job on this node ${SLURM_JOB_CPUS_PER_NODE}"
echo "Number of cpus requested per task ${SLURM_CPUS_PER_TASK}"
echo "Memory requsted per node ${SLURM_MEM_PER_NODE}"
echo "Memory requested per cpu ${SLURM_MEM_PER_CPU}"
echo "Node ID allocates ${SLURM_JOB_NODELIST}"

basedir="/scratch/sclab/100322_ATAC"
cd $basedir
echo "here i am !! $PWD"

#### get fragment distribution summary ####
#para: --cpus-per-task=8 --mem-per-cpu=4G --time=01:00:00 for 20 bam files
#conda run -n specseq Rscript fragment_size_distri.R

#### run differential peak analysis ####
#para: --mem-per-cpu=8G --cpus-per-task=16 --exclusive --time=04:00:00 for 20 atac samples 
#!!only takes like 30min for 9 samples on the new login node
#conda run -n specseq Rscript analyze_diffpeak_atac.R $basedir ${SLURM_CPUS_PER_TASK}

#### fetch fasta sequences ####
# indexColumn <- "peak.id"
# inputBed <- "/mnt/v/yqzheng/qiaoer/VSCode_yiqiao/Yiqiao_ATAC/100322_ATAC/vscode_temp/data/atac_clustered_peakset/khom/khom_gain_regions.bed"
# outputFasta <- "/mnt/v/yqzheng/qiaoer/VSCode_yiqiao/Yiqiao_ATAC/100322_ATAC/ATACQC/difffasta/atac_clustered_peaks/khom/khom_gain_regions.fa"

#echo "Fetching fasta sequences for ${inputBed}"
#echo "Will write to ${outputFasta}"
#conda run -n mpra Rscript coordinate_to_fasta.R ${inbed} ${outtsv} ${outrds}

#### great annotation ####
#genotypes=(wt at aa nt nn ww)
#genotype=${genotypes[${SLURM_ARRAY_TASK_ID}]}
#inbed="${basedir}/vscode_temp/data_new/atac_consensus_peakset/${genotype}_relax_consensus_peaks.tsv"
#outtsv="${basedir}/great/${genotype}_relax_consensus_peaks_GREATgenes.tsv"
#outrds="${basedir}/great/${genotype}_relax_consensus_peaks_GREATgenes.rds"

#genotype="e80a"
#inbed="${basedir}/vscode_temp/data_new/atac_consensus_peakset/100322_atac_${genotype}_readsInPeakNormed.consensus.centered.tsv"
#outtsv="${basedir}/great/100322_atac_${genotype}_readsInPeakNormed.consensus.centered_GREATgenes.tsv"
#outrds="${basedir}/great/100322_atac_${genotype}_readsInPeakNormed.consensus.centered_GREATgenes.rds"

#inbed="/scratch/sclab/Aldiri_ATAC/vscode_temp/data/aldiri_atac_allPeaks_notCentered.tsv"
#outtsv="/scratch/sclab/Aldiri_ATAC/vscode_temp/data/aldiri_atac_allPeaks_notCentered_GREATgenes.tsv"
#outrds="/scratch/sclab/Aldiri_ATAC/vscode_temp/data/aldiri_atac_allPeaks_notCentered_GREATgenes.rds"

#echo "Submmiting ${inbed} for GREAT analysis"
#echo "Will write to ${outtsv}"
#conda run -n mpra Rscript great_analysis_fromBED.R ${inbed} ${outtsv} ${outrds}

echo "ha! tihs is the end of the script!"

