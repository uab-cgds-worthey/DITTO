# Variant annotation

Annotated variants in VCF using Variant Effect Predictor (VEP).

Script [`src/run_pipeline.sh`](src/run_pipeline.sh) runs the snakemake workflow, which sets up VEP and then uses it for annotation.

## Setup

1. Download submodules

```sh
git submodule update --init
```


2. Create necessary directories to store log files

```sh
cd variant_annotation
mkdir -p logs/rule_logs
```

3. Create dataset config YAML and populate with paths

```sh
touch ~/.ditto_datasets.yaml
```

Enter path info into the YAML file in the following format

```yml
cadd_snv: "/path/to/data/cadd/hg38/v1.6/whole_genome_SNVs.tsv.gz"
cadd_indel: "/path/to/data/cadd/raw/hg38/v1.6/gnomad.genomes.r3.0.indel.tsv.gz"
gerp: "/path/to/data/gerp/processed/hg38/v1.6/gerp_score_hg38.bg.gz"
gnomad_genomes: "/path/to/data/gnomad/v3.0/data/gnomad.genomes.r3.0.sites.vcf.bgz"
clinvar: "/path/to/data/clinvar/data/grch38/20210119/clinvar_20210119.vcf.gz"
dbNSFP: "/path/to/data/dbnsfp/processed/v4.1a_20200616/dbNSFP4.1a_variant.complete.bgz"
```

## Datasets in custom format

Two of the datasets listed in the datasets YAML require custom formatting for use with the VEP annotator. The following
describes that formatting process that will need to be performed.

**gerp:**

 - GERP is extracted from the annotation database distributed by CADD found [here](https://cadd.gs.washington.edu/download)
 - Format GERP base-wise RS scores from extracted annotation file into final compressed BedGraph file

**dbNSFP:**

 - dbNSFP data is extracted from dbNSFP zip found [here](https://sites.google.com/site/jpopgen/dbNSFP)
 - per chromosome tab-seperated value files are extracted from the zip, sorted by GRCh38/hg38 coordinates, joined
 into a single file, bgzipped and indexed.

All other dataset files listed in the config file are in usable in the format provided by their originating source.

## How to run

- To run in current session (Note: only runs main Snakemake process in current session, Snakemake will still send jobs
 to Slurm):

    ```sh
    cd variant_annotation
    ./src/run_pipeline.sh -v .test/data/raw/testing_variants_hg38.vcf -o .test/data/processed/vep -d ~/.ditto_datasets.yaml
    ```

- To run it as slurm job:

    ```sh
    cd variant_annotation
    ./src/run_pipeline.sh -s -v .test/data/raw/testing_variants_hg38.vcf -o .test/data/processed/vep -d ~/.ditto_datasets.yaml
    ```
