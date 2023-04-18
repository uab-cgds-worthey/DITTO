#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python slurm-launch.py --exp-name testing --command "python Ditto/filter.py -i ../data/processed/testing/CAGI6_RGP_TRAIN_12_PROBAND_vep-annotated_filtered.tsv -O ../data/processed/testing/CAGI6_RGP_TRAIN_12_PROBAND"

import pandas as pd

pd.set_option("display.max_rows", None)
import numpy as np
from tqdm import tqdm
import seaborn as sns
import yaml
import os
import argparse
import matplotlib.pyplot as plt

# from sklearn.linear_model import LinearRegression
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# import pickle


def get_col_configs(config_f):
    with open(config_f) as fh:
        config_dict = yaml.safe_load(fh)

    # print(config_dict)
    return config_dict


def extract_col(config_dict, df, stats):
    print("Extracting columns and rows according to config file !....")
    df = df[config_dict["columns"]]
    if "non_snv" in stats:
        # df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
        df = df[
            (df["Alternate Allele"].str.len() > 1)
            | (df["Reference Allele"].str.len() > 1)
        ]
        print("\nData shape (non-snv) =", df.shape, file=open(stats, "a"))
    else:
        # df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
        df = df[
            (df["Alternate Allele"].str.len() < 2)
            & (df["Reference Allele"].str.len() < 2)
        ]
        if "protein" in stats:
            df = df[df["BIOTYPE"] == "protein_coding"]
        else:
            pass
        print("\nData shape (snv) =", df.shape, file=open(stats, "a"))
    df = df.loc[df["Consequence"].isin(config_dict["Consequence"])]
    print("\nData shape (nsSNV) =", df.shape, file=open(stats, "a"))

    # print('\nhgmd_class:\n', df['hgmd_class'].value_counts(), file=open(stats, "a"))
    print(
        "\nclinvar_CLNSIG:\n",
        df["clinvar_CLNSIG"].value_counts(),
        file=open(stats, "a"),
    )
    print(
        "\nclinvar_CLNREVSTAT:\n",
        df["clinvar_CLNREVSTAT"].value_counts(),
        file=open(stats, "a"),
    )
    print("\nConsequence:\n", df["Consequence"].value_counts(), file=open(stats, "a"))
    print("\nIMPACT:\n", df["IMPACT"].value_counts(), file=open(stats, "a"))
    print("\nBIOTYPE:\n", df["BIOTYPE"].value_counts(), file=open(stats, "a"))
    # df = df.drop(['CLNVC','MC'], axis=1)
    # CLNREVSTAT, CLNVC, MC
    return df


def fill_na(df, config_dict, column_info, stats):  # (config_dict,df):

    var = df[config_dict["var"]]
    df = df.drop(config_dict["var"], axis=1)
    print("parsing difficult columns......")
    # df['GERP'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['GERP'].str.split('&')]
    if "nssnv" in stats:
        # df['MutationTaster_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['MutationTaster_score'].str.split('&')]
        # df['MutationAssessor_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['MutationAssessor_score'].str.split('&')]
        # df['PROVEAN_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['PROVEAN_score'].str.split('&')]
        # df['VEST4_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['VEST4_score'].str.split('&')]
        # df['FATHMM_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['FATHMM_score'].str.split('&')]
        # else:
        for col in tqdm(config_dict["col_conv"]):
            df[col] = [
                np.mean(
                    [
                        float(item.replace(".", "0")) if item == "." else float(item)
                        for item in i.split("&")
                    ]
                )
                if "&" in str(i)
                else i
                for i in df[col]
            ]
            df[col] = df[col].astype("float64")

    print("One-hot encoding...")
    df = pd.get_dummies(df, prefix_sep="_")
    print(df.columns.values.tolist(), file=open(column_info, "w"))

    # lr = LinearRegression()
    # imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print("Filling NAs ....")
    # df = imp.fit_transform(df)
    # df = pd.DataFrame(df, columns = columns)

    df1 = pd.DataFrame()

    for key in tqdm(config_dict["nssnv_median_3_0_1"]):
            if key in df.columns:
                df1[key] = df[key].astype("float64")
            else:
                df1[key] = config_dict["nssnv_median_3_0_1"][key]

    #save data for SHAP
    #df = df[config_dict["nssnv_median_3_0_1"]]
    df1 = pd.concat([var.reset_index(drop=True), df1], axis=1)
    df1.to_csv("pre_processed_data.csv", index=False)
    df1 = df1.drop(config_dict["var"], axis=1)


    for key in tqdm(config_dict["nssnv_median_3_0_1"]):
        df1[key] = (
                    df1[key]
                    .fillna(config_dict["nssnv_median_3_0_1"][key])
                    .astype("float64")
                )

    df = df1
    # df = df.drop(df.std()[(df.std() == 0)].index, axis=1)
    del df1
    df = df.reset_index(drop=True)
    print(df.columns.values.tolist(), file=open(column_info, "a"))

    fig = plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(), fmt=".2g", cmap="coolwarm")  # annot = True,
    plt.savefig(f"correlation_plot.pdf", format="pdf", dpi=1000, bbox_inches="tight")

    # df.dropna(axis=1, how='all', inplace=True)
    # df['ID'] = [f'var_{num}' for num in range(len(df))]
    print("NAs filled!")
    df = pd.concat([var.reset_index(drop=True), df], axis=1)
    return df


def main(df, config_dict, stats, column_info, null_info):

    print("\nData shape (Before filtering) =", df.shape, file=open(stats, "w"))
    df = extract_col(config_dict, df, stats)
    print("Columns extracted! Extracting class info....")
    df.isnull().sum(axis=0).to_csv(null_info)
    # df.drop_duplicates()
    df.dropna(axis=1, how="all", inplace=True)
    df = fill_na(df, config_dict, column_info, stats)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", "-O", type=str, required=True, help="File path to output directory"
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input file with path"
    )

    args = parser.parse_args()

    config_f = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/testing.yaml"
    # read QA config file
    config_dict = get_col_configs(config_f)
    print("Config file loaded!")

    print("Loading data...")
    var_f = pd.read_csv(args.input, sep="\t", usecols=config_dict["columns"])
    print("Data Loaded !....")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    os.chdir(args.out_dir)
    stats = "stats_nssnv.csv"
    # print("Filtering "+var+" variants with at-least 50 percent data for each variant...")
    column_info = "columns.csv"
    null_info = "Nulls.csv"
    df = main(var_f, config_dict, stats, column_info, null_info)

    print("\nData shape (After filtering) =", df.shape, file=open(stats, "a"))
    print("writing to csv...")
    df.to_csv("processed_data.csv", index=False)
    del df
