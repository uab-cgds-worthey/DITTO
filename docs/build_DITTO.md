# DITTO

:fire: DITTO (inspired by pokemon) is a tool for exploring any type of small genetic variant and their predicted functional
impact on transcript(s).

:fire: DITTO uses an explainable neural network model to predict the functional impact of variants and utilizes SHAP to
explain the model's predictions.

:fire: DITTO provides annotations from OpenCravat, a tool for annotating variants with information from multiple data
sources.

:fire: DITTO is currently trained on variants from ClinVar and is not intended for clinical use.

## System Requirements

*OS:*

 Currently tested only in Cheaha (UAB HPC). Docker versions may need to be explored later to make it useable in Mac and Windows.

*Tools:*

- Anaconda3
- OpenCravat-2.4.1
- Git

*Resources:*

- CPU: > 2
- Storage: ~1TB (includes annotation databases from OpenCravat)
- RAM: ~50GB

> ***NOTE:*** We used 10 CPU cores, 50GB memory for training DITTO. The tuning and training process took ~16 hrs. Since
> DITTO uses tensorflow architecture, this process can be potentially accelerated using GPUs.

## Installation

### Requirements

- DITTO repo from GitHub
- OpenCravat with databases to annotate

To fetch DITTO source code, change in to directory of your choice and run:

```sh
git clone https://github.com/uab-cgds-worthey/DITTO.git
```

Create environment and install dependencies

```sh
# create conda environment. Needed only the first time.
conda env create --file configs/envs/environment.yaml

# if you need to update existing environment
conda env update --file configs/envs/environment.yaml

# activate conda environment
conda activate training
```

### Setup OpenCravat (ignore if already installed)

Please follow the steps mentioned in [install_openCravat.md](../docs/install_openCravat.md).

## Data

Download the latest clinVar variants: [Download VCF](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz)

## Annotation

```sh
oc run clinvar.vcf.gz -l hg38 -t csv --package mypackage -d path/to/output/directory/
```

> ***NOTE:*** By default OpenCravat uses all available CPUs. Please specify the number of CPU cores using this parameter
> in the above command `--mp 2`. Minimum number of CPUs to use is 2. Also, please make sure to setup `mypackage` from
> `configs` directory to your modules directory. To learn more about it, please review [OpenCravat's package](https://open-cravat.readthedocs.io/en/latest/Package.html).

## Preprocessing

By default, OpenCravat annotates all transcript level annotations for each variant in a single row. DITTO makes
transcript level predictions for each variant. To parse out each transcript level annotations to different rows, use
the below command

```sh
python src/annotation_parsing/parse_single_sample.py -i clinvar.vcf.gz.variant.csv -e parse \
-o clinvar.vcf.gz.variant.csv_parsed.csv.gz -c configs/opencravat_train_config.json
```

<!-- markdown-link-check-disable -->
Filter and process the annotations as shown in this [python
notebook](../src/annotation_parsing/opencravat_clinvar_filtering_80-20-20.ipynb). This will output training and testing
data to train the neural network.
<!-- markdown-link-check-enable -->

## Tune and Train DITTO

The below script uses the training data to tune the neural network by splitting it to train and validation data. It then
uses the testing data to calculate accuracy, roc, and prc metrics along with a SHAP plot showing the top features used
to train the model. Please modify the data path accordingly or use the published training and testing data from this repo.

```sh
python training/NN.py --train_x /data/train_class_data_80.csv.gz \
--test_x /data/test_class_data_20.csv.gz -c configs/col_config.yaml -o /data/
```

This script took 10 CPU cores, 100 GB memory and ~17 hrs to tune and train DITTO.

## Adding more databases (features) to DITTO

Follow the below steps to install and add more databases for annotation and before training:

1. Install the annotator/database using OpenCravat.

2. Add the annotator to `mypackage/mypackage.yml` and reannotate the clinvar VCF file.

3. Add the annotator to the [train config](../configs/opencravat_train_config.json) and specify how to parse the
   annotation.

4. Follow the steps from Preprocessing above to parse, filter, process, tune and train DITTO.

## Benchmarking
<!-- markdown-link-check-disable -->
Please follow the [python notebook](../src/analysis/opencravat_latest_benchmarking-Consequence_80_20.ipynb) to benchmark
DITTO with other pathogenicity predition tools. It also has code snippets to generate testing metrics and SHAP plots.
<!-- markdown-link-check-enable -->
