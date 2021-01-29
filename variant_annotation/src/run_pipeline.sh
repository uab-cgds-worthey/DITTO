#!/usr/bin/env bash
#SBATCH --job-name=vep
#SBATCH --output=logs/vep-%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=medium


usage() {
    echo "usage: $0"
    echo ""
    echo "required:"
    echo "    -v | --vcf [file]           path to VCF file to annotate"
    echo "    -o | --out [dir]           path to directory to place annotated VCF file"
    echo "    -d | --datasets [file]   path to datasets config YAML file"
    echo ""
    echo "options:"
    echo "    -s | --slurm             flag to indicate execution should be done as Slurm job"
    echo "    -h | --help              print usage info"
}

while [ "$1" != "" ]; do
    case $1 in
    -v | --vcf)
        shift
        INPUT_VCF=$1
        ;;
    -d | --datasets)
        shift
        DATASETS_CONFIG=$1
        ;;
    -o | --out)
        shift
        OUT_DIR=$1
        ;;
    -s | --slurm)
        USE_SLURM="yes"
        ;;
    -h | --help)
        usage
        exit
        ;;
    *)
        usage
        exit 1
        ;;
    esac
    shift
done

# ensure required info set either from CLI or as environement variables (when executed by slurm)
if [[ -z $INPUT_VCF || -z $DATASETS_CONFIG || -z $OUT_DIR ]]; then
    echo "Missing required input, check usage"
    usage
    exit 1
fi

module reset

if [[ -z $USE_SLURM ]]; then
    # run in current environment
    module load Anaconda3/2020.02
    module load snakemake/5.9.1-foss-2018b-Python-3.6.6
    snakemake \
        --snakefile "src/Snakefile" \
        --config vcf="${INPUT_VCF}" datasets="${DATASETS_CONFIG}" outdir="${OUT_DIR}" \
        --use-conda \
        --profile 'configs/snakemake_slurm_profile/{{cookiecutter.profile_name}}' \
        --cluster-config 'configs/cluster_config.json' \
        --cluster 'sbatch --ntasks {cluster.ntasks} --partition {cluster.partition} --cpus-per-task {cluster.cpus-per-task} --mem {cluster.mem} --output {cluster.output} --error {cluster.error} --parsable'
else
    # execute as slurm job
    module load gcc
    module load slurm
	sbatch --export=ALL,INPUT_VCF="${INPUT_VCF}",DATASETS_CONFIG="${DATASETS_CONFIG}",OUT_DIR="${OUT_DIR}" $0
fi



