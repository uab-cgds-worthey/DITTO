# openCravat
Documentation can be found [here](https://open-cravat.readthedocs.io/en/latest/index.html)

## Installation

### Create conda environment
```
conda create -n opencravat

conda activate opencravat
```

### Install openCravat and necessary modules for DITTO
```
pip3 install open-cravat

oc module install-base

oc module  install aloft cadd cadd_exome cancer_genome_interpreter ccre_screen chasmplus civic clingen clinpred clinvar cosmic cosmic_gene cscape dann dann_coding dbscsnv dbsnp dgi ensembl_regulatory_build ess_gene exac_gene fathmm fathmm_xf_coding funseq2 genehancer gerp ghis gnomad gnomad3 gnomad_gene gtex gwas_catalog linsight loftool lrt mavedb metalr metasvm mutation_assessor mutationtaster mutpred1 mutpred_indel ncbigene ndex ndex_chd ndex_signor omim pangalodb phastcons phdsnpg phi phylop polyphen2 prec provean repeat revel rvis segway sift siphy spliceai uniprot vest cgc cgd varity_r
```

## Moving Modules Directory

Use `oc config md` to see where modules are currently stored. To change the modules directory, copy data from the old
modules directory to the new one, then use `oc config md [new directory]` to point OpencRAVAT to the new directory.

Modules are saved in this location by default when you install them

`/home/tmamidi/.local/lib/python3.9/site-packages/cravat/modules`

Moved the modules to CGDS space on cheaha using the command below

`mv /home/tmamidi/.local/lib/python3.9/site-packages/cravat/modules
/data/project/worthey_lab/projects/experimental_pipelines/tarun/opencravat`


Change the location to use openCravat from using the below command

`oc config md /data/project/worthey_lab/projects/experimental_pipelines/tarun/opencravat/modules`

List of reporters

`oc module ls -a -t reporter`

Install reporters

`oc module install vcfreporter csvreporter tsvreporter -y`

Add package as described [here](https://open-cravat.readthedocs.io/en/latest/Package.html)

Here's the package for DITTO
`/data/project/worthey_lab/projects/experimental_pipelines/tarun/opencravat/modules/packages/mypackage/mypackage.yml`
