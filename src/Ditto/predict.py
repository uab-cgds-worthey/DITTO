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
        help="Input csv file with path for predictions",
    )
    parser.add_argument(
        "--sample",
        type=str,
        # required=True,
        help="Input sample name to showup in results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="ditto_predictions.csv",
        help="Output csv file with path",
    )
    parser.add_argument(
        "--output100",
        "-o100",
        type=str,
        default="ditto_predictions_100.csv",
        help="Output csv file with path for Top 100 variants",
    )
    # parser.add_argument(
    #    "--variant",
    #    type=str,
    #    help="Check index/rank of variant of interest. Format: chrX,101412604,C,T")
    args = parser.parse_args()

    # print("Loading data....")

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/testing.yaml"
    ) as fh:
        config_dict = yaml.safe_load(fh)

    # with open('SL212589_genes.yaml') as fh:
    #        config_dict = yaml.safe_load(fh)

    X = pd.read_csv(args.input)
    X_test = X
    # print('Data Loaded!')
    var = X_test[config_dict["ML_VAR"]]
    X_test = X_test.drop(config_dict["ML_VAR"], axis=1)
    X_test = X_test.values

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models/F_3_0_1_nssnv/StackingClassifier_F_3_0_1_nssnv.joblib",
        "rb",
    ) as f:
        clf = load(f)

    # print('Ditto Loaded!\nRunning predictions.....')

    y_score = clf.predict_proba(X_test)
    del X_test
    # print('Predictions finished!\nSorting ....')
    pred = pd.DataFrame(y_score, columns=["Ditto_Benign", "Ditto_Deleterious"])

    overall = pd.concat([var, pred], axis=1)

    # overall = overall.merge(X,on='Gene')
    del X, pred, y_score, clf
    overall.drop_duplicates(inplace=True)
    overall.insert(0, "PROBANDID", args.sample)
    overall["SD"] = 0
    overall["C"] = "*"
    overall = overall.sort_values("Ditto_Deleterious", ascending=False)
    # print('writing to database...')
    overall.to_csv(args.output, index=False)
    # print('Database storage complete!')

    overall = overall.drop_duplicates(
        subset=["Chromosome", "Position", "Alternate Allele", "Reference Allele"],
        keep="first",
    ).reset_index(drop=True)
    overall = overall[
        [
            "PROBANDID",
            "Chromosome",
            "Position",
            "Reference Allele",
            "Alternate Allele",
            "Ditto_Deleterious",
            "SD",
            "C",
        ]
    ]
    overall.columns = ["PROBANDID", "CHROM", "POS", "REF", "ALT", "P", "SD", "C"]
    overall.head(100).to_csv(args.output100, index=False, sep=":")
    del overall
