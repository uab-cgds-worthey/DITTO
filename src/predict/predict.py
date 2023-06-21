#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python src/predict/predict.py -i /data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/lovd.parsed.csv.gz -o /data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/ditto_predictions.csv -c /data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/configs/col_config.yaml -d /data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/train_data_3_star/

import pandas as pd
import yaml
import argparse
import os
from pathlib import Path
from tensorflow import keras

def test_parsing(dataframe, config_dict, clf):
    # Drop variant info columns so we can perform one-hot encoding
    # dataframe = dataframe[config_dict["raw_cols"]]
    dataframe["so"] = dataframe["consequence"]
    var = dataframe[config_dict["id_cols"]]
    dataframe = dataframe.drop(config_dict["id_cols"], axis=1)
    # dataframe = dataframe.replace(['.','-'], np.nan)

    # Perform one-hot encoding
    for key in config_dict["dummies_sep"]:
        if not dataframe[key].isnull().all():
            dataframe = pd.concat(
            (dataframe, dataframe[key].str.get_dummies(sep=config_dict["dummies_sep"][key])), axis=1
        )
    dataframe = dataframe.drop(list(config_dict["dummies_sep"].keys()), axis=1)
    dataframe = pd.get_dummies(dataframe, prefix_sep="_")

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

    df2 = df2*1
    y_score = clf.predict(df2.values)
    y_score = pd.DataFrame(y_score, columns=["DITTO"])
    y_score = round(1-y_score, 2)
    # print(y_score)
    # var["DITTO"] = y_score
    var = pd.concat([var.reset_index(drop=True), y_score.reset_index(drop=True)], axis=1)
    return var

def is_valid_output_file(p, arg):
    if os.access(Path(os.path.expandvars(arg)).parent, os.W_OK):
        return os.path.expandvars(arg)
    else:
        p.error(f"Output file {arg} can't be accessed or is invalid!")


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
        "--output",
        "-o",
        default="ditto_predictions.csv",
        help="Output csv file with path",
        type=lambda x: is_valid_output_file(parser, x),
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

    X = pd.read_csv(args.input,low_memory=False)

    clf = keras.models.load_model(f"{args.DITTO}/Neural_network")
    clf.load_weights(f"{args.DITTO}/weights.h5")

    var = test_parsing(X, config_dict, clf)
    var.to_csv(args.output, index=False)
