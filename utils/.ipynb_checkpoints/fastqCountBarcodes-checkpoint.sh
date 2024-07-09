#!/bin/bash

#unset PYTHONPATH
#source activate mpra

# Given a FASTQ file, count barcodes. This script can be called by another script with 5 arguments or used directly as a SLURM wrapper.

# Calling from another script
if [ "$#" -eq 6 ]
then
    prefix=$1
    p1AndSpacer=$2
    bcSize=$3
    bcToCrsFile=$4
    scriptPath=$5
    outputPath=$6
# Calling from SLRUM
elif [ ! -z "${SLURM_ARRAY_TASK_ID}" ]
then
    argfile=$1
    scriptPath=$2
    line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "${argfile}")
    prefix=$(echo "${line}" | cut -f1)
    p1AndSpacer=$(echo "${line}" | cut -f2)
    bcSize=$(echo "${line}" | cut -f3)
    bcToCrsFile=$(echo "${line}" | cut -f4)
    outputPath=$(echo "${line}" | cut -f5) #need to add this field to the argfile
# Else error
else
   echo "Usage: fastqCountBarcodes.sh fastqPrefix p1AndSpacer bcSize bcToCrs.txt path/to/this/script/"
   echo "\t- fastqPrefix is both the prefix for a file called fastqPrefix.fastq and the prefix for output files."
   echo "\t- p1AndSpacer is the concatenation of the p1 adapter and the spacer before the barcode."
   echo "\t- bcSize is the expected barcode size."
   echo "\t- bcToCrs.txt is a file that contains barcodes in the first field and CRS identifiers in the second field."
   echo "\nOR"
   echo "sbatch --array=1-arraySize fastqCountBarcodes.sh argumentFile.txt path/to/this/script/"
   echo "\tWhere argumentFile.txt has 4 fields per line, corresponding to fastqPrefix, p1AndSpacer, bcSize, bcToCrs.txt"
   echo "\tWhen called this way, the script runs as a SLURM job array."
   exit 1
fi

echo "Counting BC from FASTQ: ${prefix}.fastq"

# Pull lines from the FASTQ that contain at least 30 consecutive letters in the set [ACGTN] ...
grep -e "[ACGTN]\{30,\}" "${prefix}".fastq |
# ...then get the ${bcSize} [ACGT] letters immediately after the p1 and spacer, i.e. the cBC...
sed -n 's/.*'"${p1AndSpacer}"'\([ACGT]\{'"${bcSize}"'\}\).*/\1/p' |
# ...sort and count the barcodes...
sort | uniq -c |
# ...reformat to get rid of leading spacer and change the delimiter to a tab
sed -e 's/^ *//;s/ /\t/' > "${outputPath}/${prefix}".tmp
#sed -e 's/^ *//;s/ /,/' > "${outputPath}/${prefix}".tmp # for the sack of testing

echo "Parsed BC written to ${outputPath}/${prefix}.tmp"

python3 "${scriptPath}"/fastq_count_barcodes.py "${outputPath}/${prefix}" "${bcToCrsFile}"

# Clean up
#rm "${prefix}".tmp