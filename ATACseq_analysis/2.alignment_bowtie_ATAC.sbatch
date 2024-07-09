#!/bin/bash

#SBATCH --job-name=yq_bowtie
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_mapping/samtobam_%A_%a.out
#SBATCH --array=10

#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

# clear all module loaded in case of dependency conflicts
module purge

#load new modules required, using default
module use /opt/htcf/modules
module use /opt/htcf/modules-legacy
module use /opt/apps/labs/sclab/modules

module load bowtie2/2.3.5-python-3.6.5
module load samtools/1.12
module load bedtools/2.27.1
module load picard/2.21.4-java-11

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
fqdir="${basedir}concat_fq/"
trimdir="${basedir}trim/"
outputdir="${basedir}alignedbam/"
# check to make sure the index and blacklist directory are valid
bowtie2idxdir="/ref/sclab/genomes/mm10/genome"
blacklistdir="/ref/sclab/genomes/mm10.blacklist.bed.gz"

# retreive sample name from bowtie_meta.txt, one per line, format trimmed fq.gz file names
IFS=$'\n'
sample_list=($(<${basedir}bowtie_meta.txt))
sample="${sample_list[${SLURM_ARRAY_TASK_ID}]}"
r1_trim="${trimdir}${sample}_r001_val_1.fq.gz"
r2_trim="${trimdir}${sample}_r002_val_2.fq.gz"
inputlist=($r1_trim $r2_trim)

echo "all input trimmed fq.gz files ${inputlist[@]}"
echo "bowtie2 outputs will be written to alignedbam/${sample}"

# process file names for samtools cleanup after bowtie2 alignment
mkdir -p "${outputdir}${sample}"
bamdir="${outputdir}${sample}/"
outsam=${bamdir}bowtie_aligned.sam
pairmappedbam=${bamdir}paired_mapped.bam
pairmappedindex=${bamdir}paired_mapped.bam.bai
nomito=${bamdir}nomito.bam
outstat=${bamdir}picard_metric.txt
outbam=${bamdir}picard_filtered.bam
outbamblk=${bamdir}${sample}.sorted.blk.bam

# first map trimmed and filtered raw sequences to a reference genome
# mm10 bowtie2 index location: /scratch/sclab/genomes/mm10/Sequence/Bowtie2Index
#bowtie2 [options]* -x <bt2-idx> {-1 <m1> -2 <m2> | -U <r> | --interleaved <i> | --sra-acc <acc> | b <bam>} -S [<sam>]
# -t print walltime -p #of threds/processors -X maximum distance between paired reads -x directory of builded indexes and also need to specify basename
bowtie2 -t -p ${SLURM_CPUS_PER_TASK} -X 2000 --very-sensitive \
        -x $bowtie2idxdir \
        -1 ${inputlist[0]} -2 ${inputlist[1]} -S $outsam

# processing mapped sam file using samtools
# convert sam to bam format, keep only uniquely mapped and properly paired reads, sort by leftmost coordinates and index
echo "filtering reads in sam file"
samtools view -b $outsam | samtools view -f 0x2 -q 30 -bu | samtools sort -o $pairmappedbam && samtools index $pairmappedbam
samtools idxstats $pairmappedbam > ${bamdir}raw_idxstats.txt
samtools flagstat $pairmappedbam > ${bamdir}raw_flagstat.txt
echo "removing chrM and random chromosome reads"
# then remove mitochondria reads and sort, make sure to check conventions of mitochrondria notation, it is different for different programs
samtools idxstats $pairmappedbam | cut -f 1 | egrep -v chrM | xargs samtools view -bu $pairmappedbam | samtools sort -o $nomito

rm $pairmappedbam
rm $pairmappedindex

echo "marking duplicated reads with picard"
# use picard to filter duplicate reads, very memory intensive, good to limit the amount of RAM allocated with -Xmx value
# read more http://broadinstitute.github.io/picard/faq.html
java -Xmx2g -jar $PICARD MarkDuplicates I=$nomito O=$outbam M=$outstat REMOVE_DUPLICATES=true TMP_DIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

rm $nomito

echo "removing blacklist regions"
# remove blacklist regions of the genome, blacklist bed download from ENCODE, and index the output file for deeptools usage later
bedtools intersect -v -a $outbam -b $blacklistdir > $outbamblk && samtools index $outbamblk
samtools idxstats $outbamblk > ${bamdir}clean_idxstats.txt
samtools flagstat $outbamblk > ${bamdir}clean_flagstat.txt

rm $outbam

echo "${outbamblk} generated"

# clear all module loaded
module purge

echo "ha! tihs is the end of the script!"