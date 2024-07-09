#!/bin/bash

#SBATCH --job-name=yq_macs2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_peakcall/macs2_%A_%a.out
#SBATCH --array=0-20

#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
bamdir="${basedir}alignedbam/"
sieveddir="${basedir}sievedbam/"

#retrieve all sample names from lookup file
IFS=$'\n'
sample_list=($(<${basedir}bowtie_meta.txt))
sample_name="${sample_list[${SLURM_ARRAY_TASK_ID}]}"

inbam=${sieveddir}${sample_name}.allFragments.bam

#### run macs2 in conda peakcall environment ####
echo "${sample_name} input bam: ${inbam}"
echo "running macs2 under conda peakcall env"

# for paired end reads
conda run -n peakcall macs2 callpeak -t $inbam -f BAMPE -g mm \
--nomodel --keep-dup all \
-n ${sample_name}_macs2 \
--tempdir ${TMPDIR} \
--outdir ${basedir}macs2peak/${sample_name}_macs2

echo "ha! tihs is the end of the script!"
