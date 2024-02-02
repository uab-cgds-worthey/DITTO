#!/bin/bash
#
#SBATCH --job-name=DITTO
#SBATCH --output=DITTO_logs.out
#
# Number of tasks needed for this job. Generally, used with MPI jobs
#SBATCH --ntasks=1
#SBATCH --partition=amd-hdr100-res
#SBATCH --time=06:00:00
#
# Number of CPUs allocated to each task.
#SBATCH --cpus-per-task=1
#
# Mimimum memory required per allocated  CPU  in  MegaBytes.
#SBATCH --mem=10G
#
# Send mail to the email address when the job fails
#SBATCH --mail-type=FAIL

#Set your environment here
module reset
module load Java/13.0.2
module load Anaconda3
#conda activate nextflow

#Modify paths and run the pipeline here
/data/project/worthey_lab/tools/nextflow/nextflow-22.10.7/nextflow run ../pipeline.nf \
  --outdir /data/results \
  -work-dir .work_dir/ \
  --build hg38 -c cheaha.config -with-report \
  --sample_sheet .test_data/file_list.txt -resume

#https://training.nextflow.io/basic_training/cache_and_resume/#how-to-organize-in-silico-experiments
