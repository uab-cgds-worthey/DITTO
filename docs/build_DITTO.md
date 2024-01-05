# DITTO

:fire: DITTO (inspired by pokemon) is a tool for exploring any type of genetic variant and their predicted functional impact on transcript(s).
:fire: DITTO uses an explainable neural network model to predict the functional impact of variants and utilizes SHAP to explain the model's predictions.
:fire: DITTO provides annotations from OpenCravat, a tool for annotating variants with information from multiple data sources.
:tada: DITTO is currently trained on variants from ClinVar and is not intended for clinical use.

![GIF Placeholder](https://media.giphy.com/media/pMFmBkBTsDMOY/giphy.gif)

## System Requirements

*OS:*

 Currently tested only in Cheaha (UAB HPC). Docker versions may need to be explored later to make it useable in Mac and Windows.

*Tools:*

- Anaconda3 or pip3
- OpenCravat
- Git

*Resources:*

- CPU: > 2
- Storage: ~1TB (includes annotation databases from OpenCravat)
- RAM: ~50GB

> **_NOTE:_** We used 10 CPU cores, 50GB memory for training DITTO. The tuning and training process took ~16 hrs. Since
> DITTO uses tensorflow architecture, this process can potentially be accelerated using GPUs.


## Installation

### Requirements:

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

Please follow the steps mentioned in [install_openCravat.md](docs/install_openCravat.md).


