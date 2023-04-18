#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

pd.set_option("display.max_rows", 200)
import numpy as np

# from tqdm import tqdm
import yaml
import os

# from sklearn.linear_model import LinearRegression
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
import sys


def get_col_configs(config_f):

    with open(config_f) as fh:
        config_dict = yaml.safe_load(fh)

    # print(config_dict)
    return config_dict


def filter(config_dict, df):
    print("Extracting columns and rows according to config file !....")
    df = df[config_dict["columns"]]
    var = df[["AAChange.refGene", "ID"]]
    df = df.drop(["AAChange.refGene", "ID", "CLNSIG"], axis=1)
    df.replace(".", np.nan, inplace=True)
    # df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df.to_csv("./data/interim/temp1.csv", index=False)
    df = pd.read_csv("./data/interim/temp1.csv")
    os.remove("./data/interim/temp1.csv")
    df = pd.get_dummies(df, prefix_sep="_")
    df1 = pd.DataFrame()
    for key in config_dict["Fill_NAs"]:
        if key in df.columns:
            df1[key] = df[key]
        else:
            df1[key] = 0

    df = df1
    del df1
    # columns = df.columns
    # lr = LinearRegression()
    # imp= IterativeImputer(estimator=lr, verbose=2, max_iter=3, tol=1e-10, imputation_order='roman')
    print("Filling NAs using median values...")
    # df1 = imp.fit_transform(df)
    # filehandler = open('./data/processed/imputer.pkl', 'wb')
    # pickle.dump(imp, filehandler)
    # df = pd.DataFrame(df, columns = columns)
    df["is_snp"] = df["is_snp"].map({False: 0, True: 1})
    df = df.fillna(df.median())
    # for key in tqdm(config_dict['Fill_NAs']):
    #    if key in df.columns:
    #        df1[key] = df[key].fillna(config_dict['Fill_NAs'][key]).astype('float64')
    #    else:
    #        df1[key] = config_dict['Fill_NAs'][key]
    print("NAs filled!")
    for i in df.columns:
        if type(df[i]) == np.object:
            print(i)
            sys.exit("object columns identified")
    df = pd.concat([var.reset_index(drop=True), df], axis=1)
    return df


def main(var_f, config_f):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print("Config file loaded!\nNow loading data.....")
    # read clinvar data
    df = pd.read_csv(var_f)
    print("Data Loaded!")
    df = filter(config_dict, df)
    print("Data filtered!")
    # df.isnull().sum(axis = 0).to_csv('./data/processed/SL135596-NA-count-6-class.csv')
    # y = df.CLNSIG.str.replace(r'/Likely_pathogenic','').str.replace(r'/Likely_benign','')
    # y = y.str.replace(r'Likely_benign','Benign').str.replace(r'Likely_pathogenic','Pathogenic')
    # df = df.drop('CLNSIG', axis=1)

    # print dataframe shape
    # df.dtypes.to_csv('../../data/interim/head.csv')
    print("Data shape=", df.shape)
    # print('Class shape=', y.shape)
    df.to_csv("./data/processed/sample1-filtered.csv", index=False)  # sample1-filtered.
    # y.to_csv('./data/processed/sample1-y.csv', index=False)
    return None


if __name__ == "__main__":

    os.chdir("/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/")
    var_f = "./data/interim/filtered_sample.csv"
    config_f = "./configs/col_config.yaml"

    main(var_f, config_f)
