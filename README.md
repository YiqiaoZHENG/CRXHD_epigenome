# Aberrant homeodomain-DNA cooperative dimerization underlies distinct developmental defects in two dominant _CRX_ retinopathy models
This repository contains all codes and selected processed data necessary to reproduce figures in [Zheng et al.](https://pubmed.ncbi.nlm.nih.gov/38559186/)

## Prerequisites and Data availability
Necessary software packages with versions used to process data are described in text file software_versions.txt
- Coopseq data mapping and quantification were performed on [WSL:Ubuntu-20.04](https://docs.microsoft.com/en-us/windows/wsl/).
- ATACseq data mapping and quantification were performed on the WashU High Throughput Computing Facility ([HTCF](https://htcf.wustl.edu/docs/)) using [SLURM](https://slurm.schedmd.com/documentation.html).
- Further processing of intermediate data and visualization of processed data was performed on the WashU High Throughput Computing Facility.
- Raw data and additional processed data for this manuscript can be downloaded from GEO under SuperSeries [GEO:GSE256215](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE256215).

## Repository organization
- `ATACseq_analysis`, `Coopseq_analysis`, `Specseq_analysis` each contains scripts, metadata text files, and any intermediate data generated under the `processed_data` folder. Detailed descriptions of the usage of these scripts can be found in the README.md under each sub-directory.
- `1.manuscript_figures_epi.ipynb` is used to generate figures and perform statistics with processed data from all three datasets. The notebook follows the figure orders in the manuscript.
- `Figures` contains all main figures and extended data figures for the manuscript generated by the Jupyter notebook `1.manuscript_figures.ipynb`.
- `utils` contains files with Python functions used to process intermediate data and to visualize the processed data.

## Citation
doi: 10.1101/gr.279340.124
    
