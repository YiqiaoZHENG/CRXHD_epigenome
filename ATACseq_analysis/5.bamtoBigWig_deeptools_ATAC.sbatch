#!/bin/bash

#SBATCH --job-name=yq_deeptools
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_deeptools/bigwigAvg_%A_%a.out
#SBATCH --array=10

#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
#TMPDIR="/tmp/.yiqiao.zheng/${SLURM_JOB_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

basedir="/scratch/sclab/100322_ATAC"
cd $basedir
echo "here i am !! $PWD"

#directory variables
sieveddir="${basedir}/sievedbam/"
bigwigdir="${basedir}/bigwigs/"

#### preprocessings for bamCoverage to convert bam to bigwig format ####
#retrieve all sample names from file look_up.txt, one sample name/line and stored by SLURM array ID
IFS=$'\n'
sample_list=($(<$basedir/bowtie_meta.txt))
sample=${sample_list[${SLURM_ARRAY_TASK_ID}]}

# process file names for bamCoverage
allFrag="${sieveddir}${sample}.allFragments.bam"
coverage=${bigwigdir}${sample}.sorted.blk.bigWig

# convert filtered, sorted, indexed bam file to bigwig
conda run -n liftover bamCoverage -b $outbamblk -o $coverage --binSize 10 -e --normalizeUsing CPM -p ${SLURM_CPUS_PER_TASK}

#### end of bamCoverage ####


echo "ha! tihs is the end of the script!"
