# Variant annotation

Annotated variants in VCF using Variant Effect Predictor (VEP).

Script [`src/run_pipeline.sh`](src/run_pipeline.sh) runs the snakemake workflow, which sets up VEP and then uses it for annotation.


## How to run

* To run the pipeline locally (Note: Snakemake will still send jobs to cluster):

    ```sh
    cd variant_annotation
    ./src/run_pipeline.sh
    ```

* To run the pipeline as slurm job:

    ```sh
    cd variant_annotation
    sbatch src/run_pipeline.sh
    ```
