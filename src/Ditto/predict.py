#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load
import argparse

if __name__ == "__main__":
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
            default="predictions.csv",
            help="Output csv file with path")
        parser.add_argument(
            "--output500",
            "-o5",
            type=str,
            default="predictions_500.csv",
            help="Output csv file with path for Top 500 variants")
        parser.add_argument(
            "--variant",
            type=str,
            help="Check index/rank of variant of interest. Format: chrX,101412604,C,T")
        args = parser.parse_args()

        variants = args.variant.split(',')
        print("Loading data....")

        with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/testing.yaml") as fh:
                config_dict = yaml.safe_load(fh)

        #with open('SL212589_genes.yaml') as fh:
        #        config_dict = yaml.safe_load(fh)

        X = pd.read_csv(args.input)
        X_test = X
        print('Data Loaded!')
        var = X_test[config_dict['ML_VAR']]
        X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
        X_test = X_test.values

        with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models/F_3_0_1_nssnv/StackingClassifier_F_3_0_1_nssnv.joblib", 'rb') as f:
            clf = load(f)

        print('Ditto Loaded!\nRunning predictions.....')

        y_score = clf.predict_proba(X_test)
        del X_test
        print('Predictions finished!\nSorting ....')
        pred = pd.DataFrame(y_score, columns = ['pred_Benign','pred_Pathogenic'])

        overall = pd.concat([var, pred], axis=1)

        overall = overall.merge(X,on='ID')
        del X, pred, y_score, clf
        overall.drop_duplicates(inplace=True)
        #overall.sort_values('pred_Benign', ascending=False).head(500).to_csv(args.output500, index=False)
        overall = overall.sort_values('pred_Pathogenic', ascending=False)
        overall = overall.reset_index(drop=True)
        
        if variants:
                print('Finding the rank....')
                genes = overall[['SYMBOL_x','Chromosome_x','Position_x','Alternate Allele_x','Reference Allele_x']]
                genes = genes.drop_duplicates().reset_index(drop=True)
                print(f"Index/Rank of {variants}: {genes.loc[(genes['Chromosome_x'] == variants[0]) & (genes['Position_x'] == int(variants[1])) & (genes['Alternate Allele_x'] == variants[3]) & (genes['Reference Allele_x'] == variants[2])].index}") 
                del genes
        overall.head(500).to_csv(args.output500, index=False)
        print('writing to database...')
        overall.to_csv(args.output, index=False)
        print('Database storage complete!')
        