#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# module load Anaconda3/2020.02
# source activate envi
# python /data/project/worthey_lab/projects/experimental_pipelines/annovar_vcf_annotation/Annovar_Tarun.py  /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/external/filtered_SL156674.vcf /data/scratch/tmamidi/ /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/interim/ /data/project/worthey_lab/tools/annovar/annovar_hg19_db

import allel

# print(allel.__version__)

import os

os.chdir("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/")
# print(os.listdir())

print("Converting vcf.....")
# df = allel.vcf_to_dataframe('./external/clinvar.out.hg19_multianno.vcf', fields='*')
df = allel.vcf_to_dataframe("./interim/SL212589.out.hg19_multianno.vcf", fields="*")
# df.head(2)
# df.SIFT_score.unique()
df["ID"] = [f"var_{num}" for num in range(len(df))]
print("vcf converted to dataframe.\nWriting it to a csv file.....")
df.to_csv("./interim/filtered_sample.csv", index=False)
print("vcf to csv conversion completed!")
# df.to_csv("./external/clinvar.out.hg19_multianno.csv", index=False)
# print(df.head(20))
