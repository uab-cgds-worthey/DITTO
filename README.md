# DITTO

***!!! For research purposes only !!!***

> **_NOTE:_**  In a past life, DITTO used a different remote Git management provider, [UAB
> Gitlab](https://gitlab.rc.uab.edu/center-for-computational-genomics-and-data-science/sciops/ditto). It was migrated to
> Github in April 2023, and the Gitlab version has been archived.


**Aim:** We aim to develop a pipeline for accurate and rapid interpretation of genetic variants for pathogenicity using patient’s genotype (VCF) information.


## Usage

### Webapp

DITTO is available for public use at this [site](https://cgds-ditto.streamlit.app/). Here's an example on how it looks
like

![Screenshot](data/webapp.png)

### Setting up to use locally

> **_NOTE:_** Currently tested only in Cheaha (UAB HPC). Docker versions may need to be explored later to make it
> useable in Mac and Windows.

#### System Requirements

*OS:*

> **_NOTE:_** Currently tested only in Cheaha (UAB HPC). Docker versions may need to be explored later to make it
> useable in Mac and Windows.

*Tools:*

- Anaconda3
- OpenCravat
- Git

*Resources:*

- CPU: > 2
- Storage: ~1TB
- RAM: ~25GB for a WGS VCF sample

#### Installation

Installation requires the following:

- DITTO repo from GitHub
- OpenCravat with databases to annotate
- Nextflow

To fetch DITTO source code, change in to directory of your choice and run:

```sh
git clone https://github.com/uab-cgds-worthey/DITTO.git
```


Create an environment via conda or pip. Below is an example to install `nextflow` and `OpenCravat` using `pipenv`:

```sh
# create environment. Needed only the first time. Please use the above link if you're not using Mac.
python -m venv ditto-env

source ditto-env/bin/activate

# Install nextflow
pip install nextflow open-cravat
```

##### Setup OpenCravat (only one-time installation)

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


## Reproducing the DITTO model


## Contact information

For issues, please send an email with clear description to

Tarun Mamidi    -   tmamidi@uab.edu
