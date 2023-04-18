#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python predictions.py -i /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/dbnsfp_genes/${gene##*/}/dbNSFP_${gene##*/}_variants.tsv.gz  --ditto /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/dbnsfp_predictions/${gene##*/}_ditto_predictions.csv.gz

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load, dump
from tqdm import tqdm
import argparse
import shap
import numpy as np
import functools
print = functools.partial(print, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input csv file with path for filtering and predictions",
    )
    parser.add_argument(
        "--ditto",
        type=str,
        default="ditto_predictions.csv.gz",
        help="Output file with path (default:ditto_predictions.csv.gz)",
    )


    args = parser.parse_args()

    print("Loading data and Ditto model....")

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/col_config.yaml"
    ) as fh:
        config_dict = yaml.safe_load(fh)


    def parse_and_predict(dataframe, config_dict, explainer):
        dataframe.columns = config_dict["raw_cols"]
        var = dataframe[config_dict['ditto_info']]
        dataframe = dataframe[config_dict["columns"]]
        # Drop variant info columns so we can perform one-hot encoding
        dataframe = dataframe.drop(config_dict['var'], axis=1)
        dataframe = dataframe.replace(['.','-'], np.nan)

        for key in tqdm(dataframe.columns):
                try:
                    dataframe[key] = (
                        dataframe[key]
                        .astype("float32")
                    )
                except:
                    dataframe[key] = dataframe[key]

        #Perform one-hot encoding
        dataframe = pd.get_dummies(dataframe, prefix_sep='_')
        dataframe[config_dict['allele_freq_columns']] = dataframe[config_dict['allele_freq_columns']].fillna(0)

        for key in tqdm(config_dict['nssnv_median'].keys()):
                if key in dataframe.columns:
                    dataframe[key] = (
                        dataframe[key]
                        .fillna(config_dict['nssnv_median'][key])
                        .astype("float32")
                    )

        df2 = pd.DataFrame()
        for key in tqdm(config_dict['nssnv_columns']):
                if key in dataframe.columns:
                    df2[key] = dataframe[key]
                else:
                    df2[key] = 0

        del dataframe


        df2 = df2.drop(config_dict['var'], axis=1)
        X_test = df2.values
        y_score = clf.predict_proba(X_test)
        del X_test
        pred = pd.DataFrame(y_score, columns=["Ditto_Benign", "Ditto_Deleterious"])

        ditto_scores = pd.concat([var, pred], axis=1)
        ditto_scores.to_csv(args.ditto, index=False,
               compression="gzip")

        del df2

        return None


    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models_custom/dbnsfp/StackingClassifier_dbnsfp.joblib",
        "rb",
    ) as f:
        clf = load(f)

    X_train = pd.read_csv('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_custom_data-dbnsfp.csv')
    X_train = X_train.drop(config_dict['var'], axis=1)
    X_train = X_train.values
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    del background, X_train


    print('Processing data...')
    df = pd.read_csv(args.input, sep='\t', header=None)

    parse_and_predict(df, config_dict, explainer)


