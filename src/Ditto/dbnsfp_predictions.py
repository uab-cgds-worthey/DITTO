#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python slurm-launch.py --exp-name Training --command "python Ditto/dbnsfp_predictions.py -i /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/all_data_custom-dbnsfp.csv -O /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/ditto_predictions.csv.gz" --partition express --mem 10G

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load, dump
import argparse
import shap
import numpy as np
import matplotlib.pyplot as plt
import functools
print = functools.partial(print, flush=True)
from sklearn.preprocessing import label_binarize
from sklearn import metrics

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
        "--output",
        "-O",
        type=str,
        default="ditto_predictions.csv.gz",
        help="Output file path (default:ditto_predictions.csv.gz)",
    )

    args = parser.parse_args()

    print("Loading data and Ditto model....")

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/col_config.yaml"
    ) as fh:
        config_dict = yaml.safe_load(fh)


    X_test = pd.read_csv(args.input)
    var = X_test[config_dict["var"]]
    X_test = X_test.drop(config_dict["var"], axis=1)
    X_test = X_test.values

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models_custom/dbnsfp/StackingClassifier_dbnsfp.joblib",
        "rb",
    ) as f:
        clf = load(f)

    print('Ditto Loaded!\nRunning predictions.....')

    y_score = clf.predict_proba(X_test)
    del X_test, clf
    pred = pd.DataFrame(y_score, columns=["Ditto_Benign", "Ditto_Deleterious"])

    overall = pd.concat([var, pred], axis=1)
    overall.to_csv(args.output, index=False,
           compression="gzip")
    del y_score, overall
