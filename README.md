# DITTO

***!!! For research purposes only !!!***

> **_NOTE:_**  In a past life, DITTO used a different remote Git management provider, [UAB
> Gitlab](https://gitlab.rc.uab.edu/center-for-computational-genomics-and-data-science/sciops/ditto). It was migrated to
> Github in April 2023, and the Gitlab version has been archived.


**Aim:** We aim to develop a pipeline for accurate and rapid prioritization of variants using patient’s genotype (VCF) and/or phenotype (HPO) information.

## Data

Input for this project is a single sample VCF file. This will be annotated using openCravat and given to Ditto for predictions.

## Usage

### Installation

Installation simply requires fetching the source code. Following are required:

- Git

To fetch source code, change in to directory of your choice and run:

```sh
git clone https://github.com/uab-cgds-worthey/DITTO.git
```

### Requirements

*OS:*

Currently works only in Linux OS. Docker versions may need to be explored later to make it useable in Mac (and
potentially Windows).

*Tools:*

- Anaconda3

### Activate conda environment (optional)

Change in to root directory and run the commands below:

```sh
# create conda environment. Needed only the first time.
conda env create -n nextflow


# activate conda environment
conda activate nextflow

# Install nextflow
conda install -c bioconda nextflow
```

### Steps to run DITTO predictions


**Note**: Current version of openCravat that we're using doesn't support "Spanning or overlapping deletions" variants i.e.
variants with `*` in `ALT Allele` column. More on these variants [here](https://gatk.broadinstitute.org/hc/en-us/articles/360035531912-Spanning-or-overlapping-deletions-allele-). These will be ignored when running the pipeline.

#### Run DITTO pipeline

Please modify the sample in the `model.job` file and submit cheaha job using the command below

`sbatch model.job`


## Contact information

For issues, please send an email with clear description to

Tarun Mamidi    -   tmamidi@uab.edu
