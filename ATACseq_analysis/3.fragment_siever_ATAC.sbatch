#!/bin/bash

#SBATCH --job-name=yq_deeptools
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_siever/indexbam_%A_%a.out
#SBATCH --array=10

#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

#load new modules required, using default
module use /opt/htcf/modules
module use /opt/htcf/modules-legacy
module use /opt/apps/labs/sclab/modules

module load samtools/1.12

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
fqdir="${basedir}raw_fq/"
trimdir="${basedir}trim/"
inputdir="${basedir}alignedbam/"
outputdir="${basedir}sievedbam/"
bigwigdir="${basedir}bigwigs/"


# process input bam file names
IFS=$'\n'
sample_list=($(<${basedir}bowtie_meta.txt))
sample="${sample_list[${SLURM_ARRAY_TASK_ID}]}"

outbamblk="${inputdir}${sample}/${sample}.sorted.blk.bam"
echo "input bam file ${outbamblk}"

#### shift fragments by transposon sequence length and split by nucleosomal length
sieveddir="${outputdir}${sample}/"
mkdir -p ${sieveddir}

unsortedAllFrag="${sieveddir}${sample}.unosrted.allFragments.bam"
allFrag="${sieveddir}${sample}.allFragments.bam"

#echo "running alignmentSieve output will be written to ${allFrag}"
# all fragments
conda run -n liftover alignmentSieve -b ${outbamblk} --ATACshift -o ${unsortedAllFrag} -p ${SLURM_CPUS_PER_TASK}
# sort and index
samtools view -bu ${unsortedAllFrag} | samtools sort -o ${allFrag} && samtools index ${allFrag} && rm ${unsortedAllFrag}

#### end of alignmentSieve ####



echo "ha! tihs is the end of the script!"

module purge