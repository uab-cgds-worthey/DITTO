#!/usr/bin/env bash
#SBATCH --job-name=predict_variant_score
#SBATCH --output=logs/predict_variant_score-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=short

set -euo pipefail

module reset
module load Anaconda3/2020.11
conda activate testing

mkdir -p "logs/rule_logs"

snakemake \
    --snakefile "workflow/Snakefile" \
    --use-conda \
    --profile 'variant_annotation/configs/snakemake_slurm_profile/{{cookiecutter.profile_name}}' \
    --cluster-config 'configs/cluster_config.json' \
    --cluster 'sbatch --ntasks {cluster.ntasks} --partition {cluster.partition} --cpus-per-task {cluster.cpus-per-task} --mem-per-cpu {cluster.mem-per-cpu} --output {cluster.output} --parsable'
