# Variant annotation

Sets up Variant Effect Predictor (VEP) and annotates variants in vcf with chosen annotations.


## How to run

Script `run.sh` sets up and runs the snakemake workflow. This installs and sets up VEP and then uses it for annotation.

* To run locally:

    `./run.sh`

* To run as slurm job:

    `sbatch run.sh`
