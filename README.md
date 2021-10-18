# DITTO

***!!! For research purposes only !!!***

- [DITTO](#ditto)
    - [Data](#data)
    - [Usage](#usage)
        - [Installation](#installation)
        - [Requirements](#requirements)
        - [Activate conda environment](#activate-conda-environment)
        - [Run VEP annotation](#run-vep-annotation)
        - [Parse VEP annotations](#parse-vep-annotations)
        - [Run DITTO predictions](#run-ditto-predictions)
        - [Unit Testing](#unit-testing)
    - [Contact information](#contact-information)

**Aim:** Diagnosis prediction tool using AI.

## Data

Input for this project is a single sample VCF file. This will be annotated using VEP and given to Ditto for predictions.

## Usage

### Installation

Installation simply requires fetching the source code. Following are required:

- Git

To fetch source code, change in to directory of your choice and run:

```sh
git clone -b master \
    git@gitlab.rc.uab.edu:center-for-computational-genomics-and-data-science/sciops/ditto.git
```

### Requirements

*OS:*

Currently works only in Linux OS. Docker versions may need to be explored later to make it useable in Mac (and
potentially Windows).

*Tools:*

- Anaconda3
    - Tested with version: 2020.02

### Activate conda environment

Change in to root directory and run the commands below:

```sh
# create conda environment. Needed only the first time.
conda env create --file configs/envs/testing.yaml

# if you need to update existing environment
conda env update --file configs/envs/testing.yaml

# activate conda environment
conda activate testing
```

### Steps to run DITTO predictions

Remove variants with `*` in `ALT Allele` column. We will work on this in our future release.

```sh
module load BCFtools

bcftools annotate  -e'ALT="*" || type!="snp"' path/to/indexed_vcf.gz -Oz -o path/to/indexed_vcf_filtered.vcf.gz
```

#### Run VEP annotation

Please look at the steps to run VEP [here](variant_annotation/README.md)


#### Parse VEP annotations

Please look at the steps to parse VEP annotations [here](annotation_parsing/README.md)


#### Filter variants for Ditto prediction

Filtering step includes imputation and one-hot encoding of columns.

```sh
python src/Ditto/filter.py -i path/to/parsed_vcf_file.tsv -O path/to/output_directory
```

Output from this step includes -

```directory
output_directory/
├── data.csv               <--- used for Ditto predictions
├── Nulls.csv - indicates number of Nulls in each column
├── stats_nssnv.csv - variant stats from the vcf
├── correlation_plot.pdf- Plot to check if any columns are directly correlated (cutoff >0.95)
└── columns.csv - columns before and after filtering step

```

#### Ditto prediction

```sh
python src/Ditto/predict.py  -i path/to/output_directory/data.csv --sample sample_name -o path/to/output_directory/ditto_predictions.csv -o100 .path/to/output_directory/ditto_predictions_100.csv
```

#### Combine with Exomiser scores

If phenotype terms are present for the sample, one could use Exomiser to rank genes and then prioritize Ditto predictions according to the phenotype.

```sh
python src/Ditto/combine_scores.py  --raw .path/to/parsed_vcf_file.tsv --sample sample_name --ditto path/to/output_directory/ditto_predictions.csv -ep path/to/exomiser_scores/directory -o .path/to/output_directory/predictions_with_exomiser.csv -o100 path/to/output_directory/predictions_with_exomiser_100.csv
```


### Cohort level analysis

Please refer to [CAGI6-RGP](https://gitlab.rc.uab.edu/center-for-computational-genomics-and-data-science/sciops/mana/mini_projects/rgp_cagi6) project for annotation.

For predictions, make necessary directory edits to the snakemake [workflow](workflow/Snakefile) and run the following command.

```sh
sbatch predict_variant_score.sh
```

## Contact information

For issues, please send an email with clear description to

Tarun Mamidi    -   tmamidi@uab.edu

