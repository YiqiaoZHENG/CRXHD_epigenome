#!/bin/bash

#SBATCH --job-name=yq_fimo
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_meme/fimo_%A.out

echo "Number of CPUS on the allocated node ${SLURM_CPUS_ON_NODE}"
echo "Count of processors available to the job on this node ${SLURM_JOB_CPUS_PER_NODE}"
#echo "Number of cpus requested per task ${SLURM_CPUS_PER_TASK}"
echo "Number of tasks requested per node ${SLURM_NTASKS_PER_NODE}"
echo "Node ID allocates ${SLURM_JOB_NODELIST}"

# configure parallel MEME
export OMPI_MCA_btl_base_warn_component_unused=0
export OMPI_MCA_opal_cuda_support=true

#directory variables
basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

fastadir="${basedir}DiffBind/difffasta/atac_consensus_peaks/"
beddir="${basedir}DiffBind/diffbed/atac_consensus_peakset/"
memedir="${basedir}meme/atac_consensus_peaks/"
fimodir="${basedir}fimo/atac_consensus_peaks/"
motif_file="${basedir}meme/atacPhotoreceptorMotif.meme"

#IFS=$'\n'
#sample_list=($(<${basedir}meme_fasta_meta.txt))
#sample=${sample_list[${SLURM_ARRAY_TASK_ID}]}
sample="e80a"
input_bed="${beddir}100322_atac_${sample}_readsInPeakNormed.consensus.centered.bed"
query_fa="${fastadir}100322_atac_${sample}_readsInPeakNormed.consensus.centered.fa"

markov_bg="${memedir}${sample}/background"
new_dir="${fimodir}${sample}/"

mkdir -p "${memedir}${sample}"
mkdir -p ${new_dir}

# add a helper function to check header and add if not present #

#echo "Input bed: ${input_bed}"
# generate fasta file from bed
#conda run -n specseq Rscript coordinate_to_fasta.R ${input_bed} "peak.id" ${query_fa}

echo "Query fasta: ${query_fa}"
# generate background 
conda run -n motif fasta-get-markov -m 1 -dna ${query_fa} ${markov_bg}

echo "FIMO motif scanning output will be written to ${new_dir}"
# run fimo and use 0-order Markov Background model generated from MEME
conda run -n motif fimo --bfile ${markov_bg} --oc ${new_dir} --verbosity 2 --thresh 1.0E-3 ${motif_file} ${query_fa}

echo "ha! tihs is the end of the script!"

