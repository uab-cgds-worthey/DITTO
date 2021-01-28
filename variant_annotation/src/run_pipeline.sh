#!/usr/bin/env bash
#SBATCH --job-name=vep
#SBATCH --output=logs/vep-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=medium

module reset
module load Anaconda3/2020.02
module load snakemake/5.9.1-foss-2018b-Python-3.6.6

snakemake \
	--snakefile "src/Snakefile" \
	--use-conda \
	--profile 'configs/snakemake_slurm_profile/{{cookiecutter.profile_name}}' \
	--cluster-config 'configs/cluster_config.json' \
	--cluster 'sbatch --ntasks {cluster.ntasks} --partition {cluster.partition} --cpus-per-task {cluster.cpus-per-task} --mem {cluster.mem} --output {cluster.output} --error {cluster.error} --parsable'


