#!/usr/bin/env bash
#SBATCH --job-name=vep
#SBATCH --output=logs/vep-%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --partition=short

module reset
module load Anaconda3/2020.02
module load snakemake/5.9.1-foss-2018b-Python-3.6.6

# snakemake --use-conda -j 4 -k -p -f
snakemake -s b37.Snakefile --use-conda -j 4 -k -p -f
