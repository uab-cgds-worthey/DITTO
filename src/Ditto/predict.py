#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import dump, load

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    "-i",
    type=str,
    required=True,
    help="Input csv file with path")
parser.add_argument(
    "--output",
    "-o",
    type=str,
    required=True,
    help="Output csv file with path")
parser.add_argument(
    "--output500",
    "-o5",
    type=str,
    required=True,
    help="Output csv file with path for Top 500 variants")
parser.add_argument(
    "--gene",
    type=str,
    help="Check index of gene of interest")
args = parser.parse_args()

#os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

print("Loading data....")

with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/testing.yaml") as fh:
        config_dict = yaml.safe_load(fh)

#with open('SL212589_genes.yaml') as fh:
#        config_dict = yaml.safe_load(fh)

X = pd.read_csv(args.input)
X_test = X
print('Data Loaded!')
#overall.loc[:, overall.columns.str.startswith('CLN')]
var = X_test[config_dict['ML_VAR']]
X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
#feature_names = X_test.columns.tolist()
X_test = X_test.values

with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models/F_3_0_1_nssnv/StackingClassifier_F_3_0_1_nssnv.joblib", 'rb') as f:
    clf = load(f)

print('Ditto Loaded!\nRunning predictions.....')

y_score = clf.predict_proba(X_test)
del X_test
print('Predictions finished!Sorting ....')
pred = pd.DataFrame(y_score, columns = ['pred_Benign','pred_Pathogenic'])

overall = pd.concat([var, pred], axis=1)

overall = overall.merge(X,on='ID')
#overall['hazel'] = X['Gene.refGene'].map(config_dict)
del X, pred, y_score, clf
overall.drop_duplicates(inplace=True)
#overall = overall.reset_index(drop=True)
#overall.sort_values('pred_Benign', ascending=False).head(500).to_csv(args.output500, index=False)
overall = overall.sort_values('pred_Pathogenic', ascending=False)
overall = overall.reset_index(drop=True)
#genes = overall['SYMBOL_x'].drop_duplicates().reset_index(drop=True)
#genes.to_csv(args.output500)
overall.head(500).to_csv(args.output500, index=False)
#overall = overall.sort_values([ 'CHROM', 'POS'])
#columns = overall.columns
print('writing to database...')
overall.to_csv(args.output, index=False)
print(f"Index of {args.gene}: {overall.loc[overall['SYMBOL_x'] == args.gene].index}")
print('Database storage complete!')
