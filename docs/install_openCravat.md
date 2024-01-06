# OpenCravat
Original documentation for OpenCravat can be found [here](https://open-cravat.readthedocs.io/en/latest/index.html).

## Installation

### Create conda environment
```sh
# create conda environment. Needed only the first time.
conda create -n opencravat

# activate environment
conda activate opencravat
```

### Install openCravat
```sh
pip3 install open-cravat
```
### Set Modules Directory

Use `oc config md` to see where modules directory is currently pointed to. To change the modules directory, use `oc
config md [new directory]` to point OpencRAVAT to the new directory.

Test it by using `oc config md` command. It should output the new modules directory.

### Install necessary modules for DITTO
```sh
oc module install-base

oc module  install aloft cadd cadd_exome cancer_genome_interpreter ccre_screen chasmplus civic clingen clinpred clinvar cosmic cosmic_gene cscape dann dann_coding dbscsnv dbsnp dgi ensembl_regulatory_build ess_gene exac_gene fathmm fathmm_xf_coding funseq2 genehancer gerp ghis gnomad gnomad3 gnomad_gene gtex gwas_catalog linsight loftool lrt mavedb metalr metasvm mutation_assessor mutationtaster mutpred1 mutpred_indel ncbigene ndex ndex_chd ndex_signor omim pangalodb phastcons phdsnpg phi phylop polyphen2 prec provean repeat revel rvis segway sift siphy spliceai uniprot vest cgc cgd varity_r
```

### Install reporter modules for DITTO

#### List available reporters

```sh
oc module ls -a -t reporter
```

#### Install reporters

```sh
oc module install vcfreporter csvreporter tsvreporter -y
```

## Setup modules package for DITTO pipeline

Package is a module which defines module installation and job parameters. To learn more about OpenCravat's package,
please click [here](https://open-cravat.readthedocs.io/en/latest/Package.html).

Here's the package for DITTO - `configs/mypackage/mypackage.yml`

Copy the package directory to the modules directory.

```sh
# Use this to check the modules directory
oc config md

# copy the package to the modules directory
cp -r configs/mypackage path/to/modules/directory/
```
