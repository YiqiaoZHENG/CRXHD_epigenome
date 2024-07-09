#!/bin/bash

#SBATCH --job-name=yq_homer
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_homer/homer_%A_%a.out


#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

basedir="/scratch/sclab/100322_ATAC/"
cd ${basedir}
echo "here i am !! $PWD"

#directory variables
fastadir="${basedir}DiffBind/difffasta/"
beddir="${basedir}DiffBind/diffbed/atac_consensus_peakset/"
homerdir="${basedir}homer/"
annotate_file="/scratch/sclab/mm10_refGene_annotation_final.tsv"

# retreive sample name from meta.txt, one per line
#IFS=$'\n'
#sample_list=($(<${basedir}homer_annotate_meta.txt))
#sample_list=(wt at aa nt nn ww)
sample="${sample_list[${SLURM_ARRAY_TASK_ID}]}"
query_bed="${beddir}${sample}.tsv"
output_bed="${homerdir}atac_clustered_peakset/${sample}_HomerAnnotation.tsv"

# annotate peaks
conda run -n motif annotatePeaks.pl ${query_bed} mm10 > ${output_bed}
# use GENCODE annotation (gtf or gff3 https://www.gencodegenes.org/mouse/release_M10.html)
# alternatively uscs annotation https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/genes/
#conda run -n motif annotatePeaks.pl ${query_bed} mm10 -ann ${annotate_file} > ${output_bed2}

echo "ha! tihs is the end of the script!"


