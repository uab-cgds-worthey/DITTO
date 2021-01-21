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



module reset
module load Anaconda3/2020.02
module load snakemake/5.9.1-foss-2018b-Python-3.6.6

snakemake \
	--snakefile "Snakefile" \
	--use-conda \
	--cluster-config '/data/project/worthey_lab/projects/experimental_pipelines/mana/small_var_caller_pipelines/dna-seq-gatk-variant-calling/adapt_to_CGDS_setup/configs/cluster_config.json' \
	--cluster 'sbatch --ntasks {cluster.ntasks} --partition {cluster.partition} --cpus-per-task {cluster.cpus-per-task} --mem {cluster.mem} --output {cluster.output} --error {cluster.error} --parsable' \


	--profile '/data/project/worthey_lab/projects/experimental_pipelines/mana/small_var_caller_pipelines/dna-seq-gatk-variant-calling/adapt_to_CGDS_setup/configs/snakemake_profile/{{cookiecutter.profile_name}}' \
