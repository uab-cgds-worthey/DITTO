#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load
import argparse
import os
import glob

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--input",
            "-i",
            type=str,
            required=True,
            help="Input csv file with path for predictions")
        parser.add_argument(
            "--exo-path",
            "-ep",
            type=str,
            #required=True,
            help="Input path to combine exomiser results")
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
        #print("Loading data....")

        with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/testing.yaml") as fh:
                config_dict = yaml.safe_load(fh)

        #with open('SL212589_genes.yaml') as fh:
        #        config_dict = yaml.safe_load(fh)

        X = pd.read_csv(args.input)
        X_test = X
        #print('Data Loaded!')
        var = X_test[config_dict['ML_VAR']]
        X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
        X_test = X_test.values

        with open("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models/F_3_0_1_nssnv/StackingClassifier_F_3_0_1_nssnv.joblib", 'rb') as f:
            clf = load(f)

        #print('Ditto Loaded!\nRunning predictions.....')

        y_score = clf.predict_proba(X_test)
        del X_test
        #print('Predictions finished!\nSorting ....')
        pred = pd.DataFrame(y_score, columns = ['pred_Benign','pred_Pathogenic'])

        overall = pd.concat([var, pred], axis=1)

        overall = overall.merge(X,on='ID')
        del X, pred, y_score, clf
        overall.drop_duplicates(inplace=True)

        if args.exo_path != "":
            #print('Reading Exomiser scores...')
            all_files = glob.glob(os.path.join(args.exo_path, "*.tsv"))
            exo_scores = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)
            exo_scores = exo_scores[['#GENE_SYMBOL', 'ENTREZ_GENE_ID', 'EXOMISER_GENE_PHENO_SCORE']]
            overall = overall.merge(exo_scores, left_on='SYMBOL_x', right_on='#GENE_SYMBOL')
            overall = overall.sort_values([ 'pred_Pathogenic', 'EXOMISER_GENE_PHENO_SCORE'], ascending=[False,False])
            genes = overall[['SYMBOL_x','Chromosome_x','Position_x','Alternate Allele_x','Reference Allele_x','EXOMISER_GENE_PHENO_SCORE', 'pred_Pathogenic']]
            genes = genes[genes['EXOMISER_GENE_PHENO_SCORE'] != 0]

        #overall.sort_values('pred_Benign', ascending=False).head(500).to_csv(args.output500, index=False)
        else:
            overall = overall.sort_values('pred_Pathogenic', ascending=False)
            genes = overall[['SYMBOL_x','Chromosome_x','Position_x','Alternate Allele_x','Reference Allele_x', 'pred_Pathogenic']]
        overall = overall.reset_index(drop=True)
        
        genes = genes.drop_duplicates(subset=['SYMBOL_x','Chromosome_x','Position_x','Alternate Allele_x','Reference Allele_x'], keep='first').reset_index(drop=True)

        if variants != "":
                #print('Finding the rank....')
                with open("Ditto_ranking.csv", 'a') as f:
                    f.write(f"{args.input}: {variants}: {((genes.loc[(genes['Chromosome_x'] == variants[0]) & (genes['Position_x'] == int(variants[1])) & (genes['Alternate Allele_x'] == variants[3]) & (genes['Reference Allele_x'] == variants[2])].index)+1).tolist()}")
    
                #print(f"Index/Rank of {variants}: {((genes.loc[(genes['Chromosome_x'] == variants[0]) & (genes['Position_x'] == int(variants[1])) & (genes['Alternate Allele_x'] == variants[3]) & (genes['Reference Allele_x'] == variants[2])].index)+1).tolist()}", ) 

        genes.head(500).to_csv(args.output500, index=False)
        #print('writing to database...')
        overall.to_csv(args.output, index=False)
        #print('Database storage complete!')
        del genes, overall
        