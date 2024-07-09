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


#### bigwigAverage - generate genotype average bigwig files ####
# select genotype replicate bigwigs
IFS=$'\n'
sample_list=($(<$basedir/deeptools_meta.txt))
start=`expr ${SLURM_ARRAY_TASK_ID} \* 3`
length=3

echo ${start}
# select 3 replicates for each genotype
score_names=("${sample_list[@]:${start}:${length}}")
score_list=("${score_names[@]/%/.sorted.ATACshifted.bigWig}")
score_files=("${score_list[@]/#/${bigwigdir}}")

# remove the replicate number
genotype=${score_names[0]%?}

outputname="${bigwigdir}${genotype}_avg.sorted.ATACshifted.bigWig"

echo "replicate bigwigs to be averaged:"
IFS=$'\n'; echo "${score_files[*]}"

echo "will output to ${outputname}"

# calculate mean coverage from replicate bigwig files
# to use the lastest deeptools 3.5.3, run in conda envs
conda run -n liftover bigwigAverage -b ${score_files[@]} --binSize 10 -p ${SLURM_CPUS_PER_TASK} -o ${outputname}

#### end of bigwigAverage ####


echo "ha! tihs is the end of the script!"
