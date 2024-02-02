#!/bin/bash
set -euo pipefail
# Filter the DITTO scores and other annotations after running the pipeline. Example tested on CAGI project

# Specify the input folder containing the CSV files
input_folder="/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/CAGI_TR/"

# Specify the output folder where filtered files will be saved
output_folder="/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/cagi_filtered/"

# Loop through all CSV.gz files in the input folder
for input_file in "$input_folder"*DITTO_scores.csv.gz; do
    # Extract the base filename (without path and extension)
    base_filename=$(basename "${input_file}" .csv.gz)

    # Define the output file path
    output_file="${output_folder}${base_filename}_filtered.csv"

    # Use zcat, awk, and redirection to filter the data and save to the output file
    zcat "${input_file}" | awk -F',' 'NR == 1 || ($NF > 0.9 && $(NF-1) < 0.00001) {print}' > "${output_file}"

    echo "Filtered ${input_file} and saved as ${output_file}"
done
