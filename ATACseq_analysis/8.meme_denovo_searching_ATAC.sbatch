#!/bin/bash

#SBATCH --job-name=yq_memechip
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=03:00:00
#SBATCH --array=18
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_meme/memechip_%A_%a.out

#set up the temporary directory
TMPDIR="/tmp/.yiqiao.zheng/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p $TMPDIR
echo "temporary working directory ${TMPDIR}"

# specify meme path everytime
export PATH=/ref/sclab/software/bin:/ref/sclab/software/libexec/meme-5.5.2:$PATH
# configure parallel MEME
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_opal_cuda_support=true
#For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
#To enable it, please set the environment variable OMPI_MCA_opal_cuda_support=true before
#launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
#mpiexec --mca opal_cuda_support 1 ...
 
#In addition, the UCX support is also built but disabled by default.
#To enable it, first install UCX (conda install -c conda-forge ucx). Then, set the environment
#variables OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" before launching your MPI processes.
#Equivalently, you can set the MCA parameters in the command line:
#mpiexec --mca pml ucx --mca osc ucx ...
#Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via UCX.
#Please consult UCX's documentation for detail.

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
fastadir="${basedir}DiffBind/difffasta/atac_clustered_peaks/"
beddir="${basedir}DiffBind/diffbed/atac_clustered_peakset/"
memedir="${basedir}meme/atac_clustered_peaks/"
refdb="/ref/sclab/data/meme/motif_databases/"
db1="${refdb}JASPAR/JASPAR2022_CORE_vertebrates_non-redundant_v2.meme"
db2="${refdb}MOUSE/uniprobe_mouse.meme"
db3="${refdb}EUKARYOTE/jolma2013.meme"
db4="${refdb}EUKARYOTE/homeodomain.meme"
db5="${refdb}MOUSE/HOCOMOCOv11_core_MOUSE_mono_meme_format.meme"

# for classic mode: retrieve the names of all fasta files to be submitted to meme
IFS=$'\n'
sample_list=($(<${basedir}meme_fasta_meta.txt))
sample=${sample_list[${SLURM_ARRAY_TASK_ID}]}
#query_fa=${fastadir}${sample}

# for each genotype, make its own directory
genotype=$(echo "${sample}" | cut -c1-2)
new_dir="${memedir}${genotype}/${sample}/"

input_bed="${beddir}${sample}_regions.bed"
query_fa="${fastadir}${sample}_regions.fa"

mkdir -p "${memedir}${genotype}/"
mkdir -p ${new_dir}


echo "Input bed: ${input_bed}"
# generate fasta file from bed
conda run -n specseq Rscript /scratch/sclab/mus.scATAC/coordinate_to_fasta.R ${input_bed} "peak.id" ${query_fa}

echo "input fasta $query_fa"
echo "meme outputs will be written to directory ${new_dir}" 

echo "submitting file ${sample} to MEME-CHIP"
echo "query $sample"
# directly through installed meme in /ref/sclab/software
meme-chip -oc $new_dir -db $db1 -db $db2 -db $db3 -db $db4 -db $db5 -dna -order 1 -ccut 0 \
-meme-mod anr -meme-nmotifs 20 -minw 6 -maxw 30 \
-meme-p ${SLURM_NTASKS_PER_NODE} \
-spamo-skip \
$query_fa

echo "ha! tihs is the end of the script!"
