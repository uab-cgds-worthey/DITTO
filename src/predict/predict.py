#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python src/predict/predict.py -i data/external/test_parse.csv -o data/interim -c configs/col_config.yaml -d model/

import pandas as pd
import yaml
import argparse
import os
from pathlib import Path
from tensorflow import keras
import numpy as np

def parse_and_predict(dataframe, config_dict, clf):
    # Drop variant info columns so we can perform one-hot encoding
    dataframe["so"] = dataframe["consequence"]
    var = dataframe[config_dict["id_cols"]]
    dataframe = dataframe.drop(config_dict["id_cols"], axis=1)
    dataframe = dataframe.replace(['.','-',''], np.nan)
    for key in dataframe.columns:
        try:
            dataframe[key] = dataframe[key].astype("float64")
        except:
            pass

    # Perform one-hot encoding
    for key in config_dict["dummies_sep"]:
        if not dataframe[key].isnull().all():
            dataframe = pd.concat(
            (dataframe, dataframe[key].str.get_dummies(sep=config_dict["dummies_sep"][key])), axis=1
        )

    dataframe = dataframe.drop(list(config_dict["dummies_sep"].keys()), axis=1)
    dataframe = pd.get_dummies(dataframe, prefix_sep="_")

    dataframe = dataframe*1
    df2 = pd.DataFrame(columns=config_dict["filtered_cols"])
    for key in config_dict["filtered_cols"]:
        if key in dataframe.columns:
            df2[key] = dataframe[key]
        else:
            df2[key] = 0
    del dataframe

    df2 = df2.drop(config_dict["train_cols"], axis=1)
    for key in list(config_dict["median_scores"].keys()):
        if key in df2.columns:
            df2[key] = df2[key].fillna(config_dict["median_scores"][key]).astype("float64")

    y_score = 1 - clf.predict(df2, verbose=0)
    y_score = pd.DataFrame(y_score, columns=["DITTO"])

    var = pd.concat([var.reset_index(drop=True), y_score.reset_index(drop=True)], axis=1)
    dataframe = pd.concat([var.reset_index(drop=True), df2.reset_index(drop=True)], axis=1)
    del df2
    return dataframe, var

def is_valid_output_file(p, arg):
    if os.access(Path(os.path.expandvars(arg)).parent, os.W_OK):
        return os.path.expandvars(arg)
    else:
        p.error(f"Output file {arg} can't be accessed or is invalid!")

def is_valid_dir(p, arg):
    if not Path(os.path.expandvars(arg)).is_dir():
        p.error(f"The folder {arg} does not exist!")
    else:
        return os.path.expandvars(arg)

def is_valid_file(p, arg):
    if not Path(os.path.expandvars(arg)).is_file():
        p.error(f"The file {arg} does not exist!")
    else:
        return os.path.expandvars(arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input csv file with path for predictions",
        type=lambda x: is_valid_file(parser, x),
        metavar="\b",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="./",
        help="Output directory for DITTO output files",
        type=lambda x: is_valid_dir(parser, x),
        metavar="\b",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        default="/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/configs/col_config.yaml",
        help="config file to use for predictions",
        type=lambda x: is_valid_file(parser, x),
        metavar="\b",
    )
    parser.add_argument(
        "--DITTO",
        "-d",
        type=str,
        #required=True,
        default="/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/train_data_3_star/",
        help="DITTO model for predictions",
        #type=lambda x: is_valid_file(parser, x),
        #metavar="\b",
    )
    args = parser.parse_args()

    # print("Loading data....")

    with open(
        args.config
    ) as fh:
        config_dict = yaml.safe_load(fh)

    #X = pd.read_csv(args.input,low_memory=False)
#
    clf = keras.models.load_model(f"{args.DITTO}/Neural_network")
    clf.load_weights(f"{args.DITTO}/weights.h5")
#
    #var = test_parsing(X, config_dict, clf)
    #var.to_csv(args.output, index=False)
    basename = str(args.input).split('.')[0]

    df = pd.read_csv(args.input, chunksize=100000)

    for i, df_chunk in enumerate(df):
        df2, ditto_scores  = parse_and_predict(df_chunk, config_dict, clf)
        # Set writing mode to append after first chunk
        mode = 'w' if i == 0 else 'a'

        # Add header if it is the first chunk
        header = i == 0
        #print('\nData shape (nsSNV) =', df2.shape)
        # Write it to a file
        df2.to_csv(args.outdir + f"/{basename}_filtered_annots.csv.gz", index=False,
            header=header,
            mode=mode,
            compression='gzip')
        ditto_scores.to_csv(args.outdir + f"/{basename}_DITTO_scores.csv.gz", index=False,
        header=header,
            mode=mode,
               compression="gzip")
        del df2, ditto_scores
