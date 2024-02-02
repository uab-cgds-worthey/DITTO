#!/bin/bash
#SBATCH --job-name=sort
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --partition=amd-hdr100
#SBATCH --time=06-06:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --array=0-23

# module reset
# module load Anaconda3/2020.02
module load BCFtools/1.12-GCC-10.2.0
# source activate training

n=$SLURM_ARRAY_TASK_ID # number of jobs in the array
FILES=(/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/sorted/*)
gene=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "${gene##*/}"

# sort, bgzip and tabix index the predictions
sort -t$'\t' -k1,1 -k2,2n -T $USER_SCRATCH /data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/all_snv/${gene##*/} >/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/sorted/${gene##*/}

# TO-DO: bgzip and tabix index the predictions
bgzip ${gene##*/}
