#!/bin/bash

#unset PYTHONPATH
#source activate mpra

# Given FASTQ files, demultiplexes samples by P1 adapters and then count barcodes in each file
if [ "$#" -ne 5 ]
then
    echo "Usage: demultiplexAndBarcodeCount.sh fastqToSample.txt spacerSeq bcSize path/to/this/script/ fastqLineNum"
    echo "\t- fastqToSample.txt is formatted as [fastq file name] [batch name (e.g. library1)] [p1ToSample.txt] [barcodeToSequence.txt] [output file directory]
    \tWhere p1ToSample.txt is formatted as [p1 sequence] [sample name (e.g. Rna1)]
    \t and barcodeToSequence.txt is a file that contains barcodes in the first field and CRS identifiers in the second field."
    echo "\t- spacerSeq is the sequence that is between the p1 adapter and cBC."
    echo "\t- bcSize is the expected size of the cBC"
    exit 1
fi

fastqToSample=$1
spacerSeq=$2
bcSize=$3
scriptPath=$4

# If SLURM_ARRAY_TASK_ID is not set, that means either (1) this script is being run locally or (2) there is no job array. In this case, it is assumed that there is only one FASTQ file to demultiplex and the FASTQ file is specified by line number in the fastq meta list
if [ -z "${SLURM_ARRAY_TASK_ID}" ]
then
    SLURM_ARRAY_TASK_ID=$5
fi

line=$(sed "${SLURM_ARRAY_TASK_ID}q;d" "${fastqToSample}")
fastq=$(echo "${line}" | cut -f1)
name=$(echo "${line}" | cut -f2)
p1ToSample=$(echo "${line}" | cut -f3)
bcToCrsFile=$(echo "${line}" | cut -f4)
outputPath=$(echo "${line}" | cut -f5)

echo $line
echo "input fastq:  $fastq"
echo "sample name:  $name"
echo "fastq to sample mapping: $p1ToSample"
echo "BC annotation file:   $bcToCrsFile"
echo "output direcotry: $outputPath"

sh "${scriptPath}"/demultiplexFastq.sh "${fastq}" "${p1ToSample}" "${spacerSeq}" "${bcSize}" "${name}" "${scriptPath}"

# For each sample in the original FASTQ file, count barcodes.
while read p1Line
do
    p1=$(echo "${p1Line}" | cut -f1)
    sample=$(echo "${p1Line}" | cut -f2)
    prefix="${name}${sample}"
    p1AndSpacer="${p1}${spacerSeq}"

    sh "${scriptPath}"/fastqCountBarcodes.sh "${prefix}" "${p1AndSpacer}" "${bcSize}" "${bcToCrsFile}" "${scriptPath}" "${outputPath}"

done < "${p1ToSample}"