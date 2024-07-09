#!/bin/bash

#SBATCH --job-name=yq_trimATAC
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_trimming/renameAndFastQC_%A_%a.out
#SBATCH --array=0-20

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

module load cutadapt/1.16-python3
module load fastqc/0.11.7-java-11
module load trimgalore/0.6.1-python-3.6.5-java-11

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
fqdir="${basedir}raw_fq/"
trimdir="${basedir}trim/"

#retrieve all sample names from file look_up.txt, one sample name/line and stored by SLURM array ID
IFS=$'\n'
sample_list=($(<${basedir}trim_meta.txt))
#sparse each line into array
IFS=',' read -a array <<< "${sample_list[${SLURM_ARRAY_TASK_ID}]}"
sample=${array[2]}

#output fastq.gz files after adapter trimmer and filtering
#note cutadapt automatically formats output fq name, rename to more sensible names
r1_trim="${trimdir}${array[0]%.fastq.gz}"
mv "${r1_trim}_val_1.fq.gz" "${trimdir}${sample}_r001_val_1.fq.gz"
r1_trim="${trimdir}${sample}_r001_val_1.fq.gz"
r2_trim="${trimdir}${array[1]%.fastq.gz}"
mv "${r2_trim}_val_2.fq.gz" "${trimdir}${sample}_r002_val_2.fq.gz"
r2_trim="${trimdir}${sample}_r002_val_2.fq.gz"

#output dir for fastqc, make sure to mkdir before running fastqc
qc1="${trimdir}fastqc_files/${sample}_fastqc"
qc2="${trimdir}fastqc_files/${sample}_fastqctrim"
mkdir -p "${trimdir}fastqc_files/"
mkdir -p $qc1
mkdir -p $qc2

for elem in $sample $r1_trim $r2_trim #$qc1 $qc2 $r1 $r2 
do
    echo $elem
done

#fastqc [-o output dir] [--(no)extract] [-f fastq|bam|sam] [-c contaminant file] seqfile1 .. seqfileN
#Note that the output directory must already exist, FastQC will not create it.
#fastqc before trim
fastqc -f fastq -o $qc1 $r1 $r2
#trim and fastqc the output files
trim_galore --fastqc_args "-f fastq -o $qc2" --gzip -o $BASEDIR/fq_qc_files --cores ${SLURM_CPUS_PER_TASK} --paired $r1 $r2
#trim_galore --gzip -o ${trimdir} --cores ${SLURM_CPUS_PER_TASK} --paired $r1 $r2
#qc after trim, does not need to call if fastqc already specified in trim_galore
#fastqc -f fastq -o $qc2 $r1_trim $r2_trim

# clear all module loaded
module purge

echo "ha! tihs is the end of the script!"