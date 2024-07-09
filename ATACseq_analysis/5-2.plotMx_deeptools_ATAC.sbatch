#!/bin/bash

#SBATCH --job-name=yq_deeptools
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sclab/100322_ATAC/slurm_outputs/slurm_array_deeptools/plotMx_%j.out

basedir="/scratch/sclab/100322_ATAC/"
cd $basedir
echo "here i am !! $PWD"

#directory variables
bigwigdir="${basedir}bigwigs/"
matrixdir="${basedir}deeptools/matrix/"
beddir="${basedir}deeptools/diffbed/atac_clustered_peakset/"
plotdir="${basedir}deeptools/plots/"


#standardize output names
name_suffix=crx_chip_atac.nn_gain_lost_peaks
output_gz=${matrixdir}${name_suffix}.gz
output_svg=${plotdir}${name_suffix}.svg
ref_point=summit


# 021420 crx chip sample average coverage
score_names=(wt e k r nl rd)
score_files=("${score_names[@]/%/_avg.bigWig}")
score_list=("${score_files[@]/#//scratch/sclab/021420_ChIP/bigwigs/}")
score_names=(wt.chip e80a.chip k88n.chip r90w.chip nrlko.chip nr2e3ko.chip)

# 100322 atac samples
score_names2=($(<${basedir}deeptools_meta.txt))
score_files2=("${score_names2[@]/%/sorted.ATACshifted.bigWig}")
score_list2=("${score_files2[@]/#/${bigwigdir}}")
score_names2=("${score_names2[@]/%/.atac}")

# use only atac data
score_files=(${score_files[@]} ${score_files2[@]})
score_list=(${score_list[@]} ${score_list2[@]})
score_names=(${score_names[@]} ${score_names2[@]})


echo "all input bigwig files in the current run"
IFS=$'\n'; echo "${score_list[*]}" 


# directly retreive file name from a matedata text file (line number starts from 1 instead of 0)
region_files=($(sed -n '1,2p' ${basedir}deeptools_bed_meta.txt))
region_list=("${region_files[@]/#/${beddir}}")
region_names=(nn.gain nn.lost) # make sure to always use array when there are multiple samples


echo "all regions in the current run"
IFS=$'\n'; echo "${region_files[*]}" # multiple regions
echo ${region_file} # single region

# sort by WT and heterozygous mutant ATAC
sort_order=(1 2)

echo "running computeMatrix, output will be written to ${output_gz}"
# generate the intermediate files of scores per genome region for plotHeatmap
conda run -n liftover computeMatrix reference-point -S ${score_list[@]} \
-R ${region_list[@]} \
-o $output_gz \
--samplesLabel ${score_names[@]} --referencePoint center -b 2000 -a 2000 -bs 10 \
--sortRegions descend --sortUsing mean --sortUsingSamples ${sort_order[@]} \
--missingDataAsZero --skipZeros \
-p ${SLURM_CPUS_PER_TASK}

# automatically format colors especially for a lot of atac samples
intensity_hm_colors=('#FFFFFF, #08306B' '#FFFFFF, #08306B' '#FFFFFF, #08306B' '#FFFFFF, #08306B' '#FFFFFF, #08306B' '#FFFFFF, #08306B')
atac_hm_color=('#FFFFFF, #1a1a1a')

# crx chip and atac colors
for ((j=1; j<=${#score_names2[@]}; j++)); do intensity_hm_colors+=(${atac_hm_color[@]}); done
echo "Total number of colors for intensity heatmap: ${#intensity_hm_colors[@]}"

# zscore lists
zmin_list=()
zmax_list=()
# minimal is always 0
for ((j=1; j<=${#score_names[@]}; j++)); do zmin_list+=0; done

# max - chip, atac, aldiri atac, different for khom
for ((j=1; j<=6; j++)); do zmax_list+=(1.0); done
for ((j=1; j<=${#score_names2[@]}; j++)); do zmax_list+=(1.0); done

echo "making deeptools heatmap, will be written to ${output_svg}"
conda run -n liftover plotHeatmap -m $output_gz -o ${plotdir}${name_suffix}.svg --dpi 600 \
--refPointLabel $ref_point \
--xAxisLabel 'dist. to summit (bp)' --labelRotation 0 \
--regionsLabel ${region_names[@]} --samplesLabel ${score_names[@]} \
--sortRegions descend --sortUsing mean --sortUsingSamples ${sort_order[@]} \
--zMin ${zmin_list[@]} --zMax ${zmax_list[@]} \
--colorList ${intensity_hm_colors[@]} --colorNumber 444 \
--whatToShow 'heatmap and colorbar' \
--boxAroundHeatmaps yes --legendLocation best \
--plotFileFormat svg

conda run -n liftover plotHeatmap -m $output_gz -o ${plotdir}${name_suffix}.png --dpi 600 \
--refPointLabel $ref_point \
--xAxisLabel 'dist. to summit (bp)' --labelRotation 0 \
--regionsLabel ${region_names[@]} --samplesLabel ${score_names[@]} \
--sortRegions descend --sortUsing mean --sortUsingSamples ${sort_order[@]} \
--zMin ${zmin_list[@]} --zMax ${zmax_list[@]} \
--colorList ${intensity_hm_colors[@]} --colorNumber 444 \
--whatToShow 'heatmap and colorbar' \
--boxAroundHeatmaps yes --legendLocation best \
--plotFileFormat png


# generate average intensity profile plots
# matplotlib palettes in hex code
#cmap_tab10=("#1f77b4" "#ff7f0e" "#2ca02c" "#d62728" "#9467bd" "#8c564b" "#e377c2" "#7f7f7f" "#bcbd22" "#17becf")
cmap_tab20=('#1f77b4' '#aec7e8' '#ff7f0e' '#ffbb78' '#2ca02c' '#98df8a' '#d62728' '#ff9896' '#9467bd' '#c5b0d5' '#8c564b' '#c49c94' '#e377c2' '#f7b6d2' '#7f7f7f' '#c7c7c7' '#bcbd22' '#dbdb8d' '#17becf' '#9edae5')


echo "making deeptools profile plot, will be written to ${plotdir}${name_suffix}_profile.png"
conda run -n liftover plotProfile -m $output_gz --outFileNameData ${plotdir}${name_suffix}_profile.tsv -o ${plotdir}${name_suffix}_profile_perGp.png --dpi 600 \
--refPointLabel $ref_point \
--startLabel "-2.0" --endLabel "2.0kb" \
--refPointLabel 'dist. to summit (bp)' --labelRotation 0 \
--regionsLabel ${region_names[@]} --samplesLabel ${score_names[@]} \
--plotType lines --colors ${cmap_tab20[@]} \
--legendLocation best --perGroup \
--plotFileFormat png

echo "ha! tihs is the end of the script!"

