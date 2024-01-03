# DITTO

***!!! For research purposes only !!!***

> **_NOTE:_**  In a past life, DITTO used a different remote Git management provider, [UAB
> Gitlab](https://gitlab.rc.uab.edu/center-for-computational-genomics-and-data-science/sciops/ditto). It was migrated to
> Github in April 2023, and the Gitlab version has been archived.


**Aim:** We aim to develop a pipeline for accurate and rapid interpretation of genetic variants for pathogenicity using patient’s genotype (VCF) information.

## Data

Input for this project is a samplesheet with VCF filepaths. They will be annotated using openCravat and given to Ditto for predictions.

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

Currently works only in Cheaha (UAB HPC). Docker versions may need to be explored later to make it useable in Mac (and
potentially Windows).

*Tools:*

- Anaconda3

### Activate conda environment (optional)

Change in to root directory and run the commands below:

```sh
# create conda environment. Needed only the first time.
conda env create -n ditto


# activate conda environment
conda activate ditto

# Install nextflow
conda install -c bioconda nextflow
```

### Steps to run DITTO predictions

#### Setup OpenCravat (only one-time installation)

Please follow the steps mentioned [here](docs/install_openCravat.md).

**Note**: Current version of OpenCravat that we're using doesn't support "Spanning or overlapping deletions" variants i.e.
variants with `*` in `ALT Allele` column. More on these variants [here](https://gatk.broadinstitute.org/hc/en-us/articles/360035531912-Spanning-or-overlapping-deletions-allele-). These will be ignored when running the pipeline.

#### Run DITTO pipeline

Please make a samplesheet with VCF files. Please make sure to edit the directories as needed.

```sh
nextflow run pipeline.nf \
  --outdir /data/processed/ \
  -work-dir ./wor_dir \
  --build hg38 -with-report \
  --oc_modules /data/opencravat/modules \
  --sample_sheet .test_data/file_list_partaa
```

##### Run on UAB cheaha
Please update the below file and submit cheaha (UAB HPC) job using the command below

`sbatch model.job`


## Contact information

For issues, please send an email with clear description to

Tarun Mamidi    -   tmamidi@uab.edu
