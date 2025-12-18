# DITTO

<!-- markdown-link-check-disable -->
[![Perform linting -
Markdown](https://github.com/uab-cgds-worthey/DITTO/actions/workflows/linting.yml/badge.svg)](https://github.com/uab-cgds-worthey/DITTO/actions/workflows/linting.yml)
<!-- markdown-link-check-enable -->

***!!! For research purposes only !!!***

> ***NOTE:***  In a past life, DITTO used a different remote Git management provider, [UAB
> Gitlab](https://gitlab.rc.uab.edu/center-for-computational-genomics-and-data-science/sciops/ditto). It was migrated to
> Github in April 2023, and the Gitlab version has been archived.

DITTO is an explainable neural network that can be helpful for accurate and rapid interpretation of small
genetic variants for pathogenicity using patient’s genotype (VCF) information.

## Getting Started

- [Prerequisites](#prerequisites)
- [Using DITTO](#using-ditto)
  - [Webapp](#webapp)
  - [API](#api)
  - [Prediction](#prediction)
    - [Local Prediction](#local-prediction)
    - [HPC Prediction with Cheaha](#hpc-prediction-with-cheaha)
- [Reproducing the DITTO model](#reproducing-the-ditto-model)
- [Download DITTO DB (Precomputed scores)](#download-ditto-db-precomputed-scores)
- [How to cite?](#how-to-cite)
- [Contact](#contact-information)

## Prerequisites

The following prerequisites are required to be installed in the target envrionment for deploying and running DITTO
prediction model.

### Tools

- [Python 3.10](https://www.python.org/) - [Install](https://www.python.org/downloads/)
  - The specified OpenCravat version requires Python 3.10
- [Anaconda3 25.7+](https://anaconda.com/) - [install](https://www.anaconda.com/docs/getting-started/anaconda/install)
- [OpenCravat 2.4.1](https://www.opencravat.org/) - [install](https://github.com/KarchinLab/open-cravat/releases/tag/2.4.1)
- [Git](https://git-scm.com/)
  - Setup with your favorite git client. Here is a [GitHub Guide](https://github.com/git-guides/install-git)
  for different platforms.
- [Nextflow 22.10.7+](https://www.nextflow.io/) - [install](https://www.nextflow.io/docs/latest/install.html)

> ***NOTE:*** Current version of OpenCravat that we're using doesn't support "Spanning or overlapping deletions"
> variants i.e. variants with `*` in `ALT Allele` column. More on these variants
<!-- markdown-link-check-disable -->
> [here](https://gatk.broadinstitute.org/hc/en-us/articles/360035531912-Spanning-or-overlapping-deletions-allele).
<!-- markdown-link-check-enable -->
> These will be ignored when running the pipeline.

### System Requirements

- CPU: >2
- RAM: ~25GB for a WGS VCF sample
- Storage: 1TB
  - The storage requirements are for hosting the OpenCravat annotators ~600GB of data required to store all annotators

## Using DITTO

DITTO scores for variants can be obtained by the below 3 ways. Webapp and API are for single variant analysis and the
local setup is for batch/bulk variant predictions.

### Webapp

<!-- markdown-link-check-disable -->
DITTO is available for public use at this [website](https://cgds-ditto.streamlit.app).
<!-- markdown-link-check-enable -->

### API

DITTO is not hosted as a public API but one can serve up locally to query DITTO scores. Please follow the instructions
in this [GitHub repo](https://github.com/uab-cgds-worthey/DITTO-API).

### Prediction

#### Installation

To fetch DITTO source code, change in to directory of your choice and run:

```sh
git clone https://github.com/uab-cgds-worthey/DITTO.git
cd DITTO
```

### Local Prediction

> ***NOTE:*** This setup will allow one to annotate a VCF sample and make DITTO predictions. Currently tested only in
> Cheaha (UAB HPC) because of resource limitations to download datasets from OpenCRAVAT.
> Docker versions may need to be explored later to make it useable in Mac and Windows.

#### Setup Steps

- ***Setup OpenCravat (only one-time installation)***

  Please follow the steps mentioned in [install_openCravat.md](docs/install_openCravat.md).

- ***Setup Nextflow***

  Create an environment via conda. Below is an example to install `nextflow`.
  
  ```sh
  # create environment. Needed only the first time. Please use the above link if you're not using Mac.
  conda create --name ditto-env

  conda activate ditto-env

  # Install nextflow
  conda install bioconda::nextflow=22.10 conda-forge::conda=23.1
  ```

- ***Sample Sheet***

  Please make a samplesheet `.test_data/file_list.txt` with VCF files (incl. path).

  Example `file_list.txt`:

  ```bash
  # Example is using MacOS home folder

  /Users/<username>/Workspace/DITTO/.test_data/oc_test_data.vcf.gz
  /Users/<username>/Workspace/DITTO/.test_data/testing_variants_hg38.vcf.gz
  ```

  This will run DITTO prediction for both vcf files in the `file_list.txt`.

- ***Run the NextFlow pipeline***

  Please make sure to edit the directory paths as needed and run the pipeline as shown below.

  ```sh
  # Note: NextFlow work directory is defined as `-work-dir` in the run command parameters
  # Note: `--output` cannot be relative, set a path nextflow can access. ex. `/tmp/DITTO/output`

  nextflow run pipeline.nf \
    -work-dir ./work_dir \
    --build hg38 -c ./configs/nextflow/local.config -with-report \
    --sample_sheet .test_data/file_list.txt \
    --oc_modules /<path-to>/opencravat/modules \
    --outdir $PWD/data/output
  ```

### HPC Prediction with Cheaha

To run on UAB cheaha, see the [installation](#installation) step to clone the DITTO repository into a Cheaha directory.

- Create a text file listing the path to VCF file(s) (1 path per line) with variants to score
  - Paths can be full absolute paths **or** relative paths (relative to the directory where the pipeline will be run from, **not** the directory where the `pipeline.nf` file is)
- See the example input file [.test_data/file_list.txt](.test_data/file_list.txt) (lists 2 testing example input VCFs) 
  for reference or as an input file for testing (default behavior of `model.job`)

```bash
/home/<username>/Workspace/DITTO/.test_data/oc_test_data.vcf.gz
/home/<username>/Workspace/DITTO/.test_data/testing_variants_hg38.vcf.gz
```

- Update `model.job` (change the `--sample_sheet` option to your input file with VCF path(s) and
  `--outdir` to the desired output location of DITTO predictions)

```sh
sbatch model.job
```

## Reproducing the DITTO model

Detailed instructions on reproducing the model is explained in [build_DITTO.md](docs/build_DITTO.md)

## Download DITTO DB (Precomputed scores)

Precomputed scores for all possible SNVs and known Indels from gnomADv3.0 in main chromosomes in hg38 reference genome
are available to download here - <https://s3.lts.rc.uab.edu/cgds-public/dittodb/dittodb.html>

## How to cite?

<!-- markdown-link-check-disable -->
Mamidi, T.K.K.; Wilk, B.M.; Gajapathy, M.; Worthey, E.A. DITTO: An Explainable Machine-Learning Model for
Transcript-Specific Variant Pathogenicity Prediction. Preprints 2024, 2024040837. <https://doi.org/10.20944/preprints202404.0837.v1>
<!-- markdown-link-check-enable -->

## Contact information

For queries, please open a GitHub issue.

For urgent queries, send an email with clear description to

|    Name      |        Email       |
|--------------|--------------------|
| Tarun Mamidi | <tmamidi@uab.edu>  |
| Liz Worthey  | <lworthey@uab.edu> |
