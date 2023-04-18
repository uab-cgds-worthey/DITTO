#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python slurm-launch.py --exp-name predictions --command "python Ditto/dbnsfp_prediction.py -i /data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.sorted.tsv.gz --filter /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/all_data_filter-dbnsfp.tsv.gz --ditto /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/dbnsfp_only_ditto_predictions.tsv.gz" --partition long --mem 10G

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load
from tqdm import tqdm
import argparse
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
        "--filter",
        type=str,
        default="filter.csv.gz",
        help="Output file with path (default:filter.csv.gz)",
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


    def parse_and_predict(dataframe, config_dict):

        dataframe.columns = config_dict["raw_cols"]
        dataframe = dataframe[config_dict["columns"]]
        # Drop variant info columns so we can perform one-hot encoding
        var = dataframe[config_dict['var']]
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
        df2 = pd.concat([var.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)

        return df2, ditto_scores


    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models_custom/dbnsfp/StackingClassifier_dbnsfp.joblib",
        "rb",
    ) as f:
        clf = load(f)

    print('Processing data...')


    df = pd.read_csv(args.input, sep='\t', header=None, chunksize=1000000)

    for i, df_chunk in enumerate(df):
        df2, ditto_scores  = parse_and_predict(df_chunk, config_dict)
        # Set writing mode to append after first chunk
        mode = 'w' if i == 0 else 'a'

        # Add header if it is the first chunk
        header = i == 0
        #print('\nData shape (nsSNV) =', df2.shape)
        # Write it to a file
        df2.to_csv(args.filter, index=False,
            header=header, sep='\t',
            mode=mode,
            compression='gzip')
        ditto_scores.to_csv(args.ditto, index=False,
        header=header,sep='\t',
            mode=mode,
               compression="gzip")
        del df2, ditto_scores

