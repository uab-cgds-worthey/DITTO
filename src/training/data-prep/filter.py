#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
#python slurm-launch.py --exp-name no_null --command "python training/data-prep/filter.py --var-tag no_null_nssnv --cutoff 1" --mem 50G

import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
from tqdm import tqdm 
=======
# python slurm-launch.py --exp-name no_null --command "python training/data-prep/filter.py --var-tag no_null_nssnv --cutoff 1" --mem 50G

import pandas as pd

pd.set_option("display.max_rows", None)
import numpy as np
from tqdm import tqdm
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
import seaborn as sns
import yaml
import os
import argparse
import matplotlib.pyplot as plt
<<<<<<< HEAD
#from sklearn.linear_model import LinearRegression
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#import pickle
=======

# from sklearn.linear_model import LinearRegression
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
# import pickle

>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

def get_col_configs(config_f):
    with open(config_f) as fh:
        config_dict = yaml.safe_load(fh)

    # print(config_dict)
    return config_dict


<<<<<<< HEAD
def extract_col(config_dict,df, stats, list_tag):
    print('Extracting columns and rows according to config file !....')
    df = df[config_dict['columns']]
    if 'non_snv' in stats:
        #df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
        df=df[(df['Alternate Allele'].str.len() > 1) | (df['Reference Allele'].str.len() > 1)]
        print('\nData shape (non-snv) =', df.shape, file=open(stats, "a"))
    else:
        #df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
        df=df[(df['Alternate Allele'].str.len() < 2) & (df['Reference Allele'].str.len() < 2)]
        if 'protein' in stats:
            df = df[df['BIOTYPE']=='protein_coding']
        else:
            pass
        print('\nData shape (snv) =', df.shape, file=open(stats, "a"))
    #df = df[config_dict['Consequence']]
    df= df.loc[df['Consequence'].isin(config_dict['Consequence'])]
    print('\nData shape (nsSNV) =', df.shape, file=open(stats, "a"))
    if 'train' in stats:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
    else:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_test'])]
    
    if 'train' in stats:
        print('Dropping empty columns and rows along with duplicate rows...')
        #df.dropna(axis=1, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[0]/4)
        df.dropna(axis=0, thresh=(df.shape[1]*list_tag[1]), inplace=True)  #thresh=(df.shape[1]*0.3),   how='all',
        df.drop_duplicates()
        df.dropna(axis=1, how='all', inplace=True)  #thresh=(df.shape[0]/4)
    print('\nhgmd_class:\n', df['hgmd_class'].value_counts(), file=open(stats, "a"))
    print('\nclinvar_CLNSIG:\n', df['clinvar_CLNSIG'].value_counts(), file=open(stats, "a"))
    print('\nclinvar_CLNREVSTAT:\n', df['clinvar_CLNREVSTAT'].value_counts(), file=open(stats, "a"))
    print('\nConsequence:\n', df['Consequence'].value_counts(), file=open(stats, "a"))
    print('\nIMPACT:\n', df['IMPACT'].value_counts(), file=open(stats, "a"))
    print('\nBIOTYPE:\n', df['BIOTYPE'].value_counts(), file=open(stats, "a"))
    #df = df.drop(['CLNVC','MC'], axis=1)
    # CLNREVSTAT, CLNVC, MC
    return df

def fill_na(df,config_dict, column_info, stats, list_tag): #(config_dict,df):

    var = df[config_dict['var']]
    df = df.drop(config_dict['var'], axis=1)
    print('parsing difficult columns......')
    df['GERP'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['GERP'].str.split('&')]
    if 'nssnv' in stats:
        df['MutationTaster_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['MutationTaster_score'].str.split('&')]
    #else:
    #    for col in tqdm(config_dict['col_conv']):
    #        df[col] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df[col].str.split('&')]
    if 'train' in stats:
        fig= plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), fmt='.2g',cmap= 'coolwarm') # annot = True, 
        plt.savefig(f"train_{list_tag[0]}/correlation_filtered_raw_{list_tag[0]}.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    print('One-hot encoding...')
    df = pd.get_dummies(df, prefix_sep='_')
    print(df.columns.values.tolist(),file=open(column_info, "w"))
    #df.head(2).to_csv(column_info, index=False)
    #lr = LinearRegression()
    #imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print('Filling NAs ....')
    #df = imp.fit_transform(df)
    #df = pd.DataFrame(df, columns = columns)
    
    if list_tag[2] == 1:
        print("Including AF columns...")
        df1=df[config_dict['gnomad_columns']]
        df1=df1.fillna(list_tag[3])

        if list_tag[4] == 1:
            df = df.drop(config_dict['gnomad_columns'], axis=1)
            df=df.fillna(df.median())
            if 'train' in stats:
                print('\nColumns:\t', df.columns.values.tolist(), file=open(stats, "a"))
                print('\nMedian values:\t', df.median().values.tolist(), file=open(stats, "a"))
=======
def extract_col(config_dict, df, stats, list_tag):
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
    # df = df[config_dict['Consequence']]
    df = df.loc[df["Consequence"].isin(config_dict["Consequence"])]
    print("\nData shape (nsSNV) =", df.shape, file=open(stats, "a"))
    if "train" in stats:
        df = df.loc[df["hgmd_class"].isin(config_dict["Clinsig_train"])]
    else:
        df = df.loc[df["hgmd_class"].isin(config_dict["Clinsig_test"])]

    if "train" in stats:
        print("Dropping empty columns and rows along with duplicate rows...")
        # df.dropna(axis=1, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[0]/4)
        df.dropna(
            axis=0, thresh=(df.shape[1] * list_tag[1]), inplace=True
        )  # thresh=(df.shape[1]*0.3),   how='all',
        df.drop_duplicates()
        df.dropna(axis=1, how="all", inplace=True)  # thresh=(df.shape[0]/4)
    print("\nhgmd_class:\n", df["hgmd_class"].value_counts(), file=open(stats, "a"))
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


def fill_na(df, config_dict, column_info, stats, list_tag):  # (config_dict,df):

    var = df[config_dict["var"]]
    df = df.drop(config_dict["var"], axis=1)
    print("parsing difficult columns......")
    # df['GERP'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['GERP'].str.split('&')]
    if "nssnv" in stats:
        #    df['MutationTaster_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['MutationTaster_score'].str.split('&')]
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
    if "train" in stats:
        fig = plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), fmt=".2g", cmap="coolwarm")  # annot = True,
        plt.savefig(
            f"train_{list_tag[0]}/correlation_filtered_raw_{list_tag[0]}.pdf",
            format="pdf",
            dpi=1000,
            bbox_inches="tight",
        )
    print("One-hot encoding...")
    df = pd.get_dummies(df, prefix_sep="_")
    print(df.columns.values.tolist(), file=open(column_info, "w"))
    # df.head(2).to_csv(column_info, index=False)
    # lr = LinearRegression()
    # imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print("Filling NAs ....")
    # df = imp.fit_transform(df)
    # df = pd.DataFrame(df, columns = columns)

    if list_tag[2] == 1:
        print("Including AF columns...")
        df1 = df[config_dict["gnomad_columns"]]
        df1 = df1.fillna(list_tag[3])

        if list_tag[4] == 1:
            df = df.drop(config_dict["gnomad_columns"], axis=1)
            df = df.fillna(df.median())
            if "train" in stats:
                print("\nColumns:\t", df.columns.values.tolist(), file=open(stats, "a"))
                print(
                    "\nMedian values:\t",
                    df.median().values.tolist(),
                    file=open(stats, "a"),
                )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
        else:
            pass
    else:
        print("Excluding AF columns...")
        if list_tag[4] == 1:
<<<<<<< HEAD
            df = df.drop(config_dict['gnomad_columns'], axis=1)
            df1=df.fillna(df.median())
            if 'train' in stats:
                print('\nColumns:\t', df.columns.values.tolist(), file=open(stats, "a"))
                print('\nMedian values:\t', df.median().values.tolist(), file=open(stats, "a"))
        else:
            df1 = pd.DataFrame()

    if 'non_nssnv' in stats:
        for key in tqdm(config_dict['non_nssnv_columns']):
            if key in df.columns:
                df1[key] = df[key].fillna(config_dict['non_nssnv_columns'][key]).astype('float64')
            else:
                df1[key] = config_dict['non_nssnv_columns'][key]
    else:
        for key in tqdm(config_dict['nssnv_columns']):
            if key in df.columns:
                df1[key] = df[key].fillna(config_dict['nssnv_columns'][key]).astype('float64')
            else:
                df1[key] = config_dict['nssnv_columns'][key]
=======
            df = df.drop(config_dict["gnomad_columns"], axis=1)
            df1 = df.fillna(df.median())
            if "train" in stats:
                print("\nColumns:\t", df.columns.values.tolist(), file=open(stats, "a"))
                print(
                    "\nMedian values:\t",
                    df.median().values.tolist(),
                    file=open(stats, "a"),
                )
        else:
            df1 = pd.DataFrame()

    if "non_nssnv" in stats:
        for key in tqdm(config_dict["non_nssnv_columns"]):
            if key in df.columns:
                df1[key] = (
                    df[key]
                    .fillna(config_dict["non_nssnv_columns"][key])
                    .astype("float64")
                )
            else:
                df1[key] = config_dict["non_nssnv_columns"][key]
    else:
        for key in tqdm(config_dict["nssnv_columns"]):
            if key in df.columns:
                df1[key] = (
                    df[key].fillna(config_dict["nssnv_columns"][key]).astype("float64")
                )
            else:
                df1[key] = config_dict["nssnv_columns"][key]
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    df = df1
    df = df.drop(df.std()[(df.std() == 0)].index, axis=1)
    del df1
    df = df.reset_index(drop=True)
<<<<<<< HEAD
    print(df.columns.values.tolist(),file=open(column_info, "a"))
    if 'train' in stats:
        fig= plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(),fmt='.2g',cmap= 'coolwarm') # annot = True, 
        plt.savefig(f"train_{list_tag[0]}/correlation_before_{list_tag[0]}.pdf", format='pdf', dpi=1000, bbox_inches='tight')
=======
    print(df.columns.values.tolist(), file=open(column_info, "a"))
    if "train" in stats:
        fig = plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), fmt=".2g", cmap="coolwarm")  # annot = True,
        plt.savefig(
            f"train_{list_tag[0]}/correlation_before_{list_tag[0]}.pdf",
            format="pdf",
            dpi=1000,
            bbox_inches="tight",
        )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

        # Create correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
<<<<<<< HEAD
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        print(f"Correlated columns being dropped: {to_drop}",file=open(column_info, "a"))

        # Drop features 
        df.drop(to_drop, axis=1, inplace=True)
        df = df.reset_index(drop=True)
    print(df.columns.values.tolist(),file=open(column_info, "a"))
    #df.dropna(axis=1, how='all', inplace=True)
    df['ID'] = [f'var_{num}' for num in range(len(df))]
    print('NAs filled!')
    df = pd.concat([var.reset_index(drop=True), df], axis=1)
    return df

def main(df, config_f, stats,column_info, null_info, list_tag):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print('Config file loaded!')
    # read clinvar data
    
    print('\nData shape (Before filtering) =', df.shape, file=open(stats, "a"))
    df = extract_col(config_dict,df, stats, list_tag)
    print('Columns extracted! Extracting class info....')
    df.isnull().sum(axis = 0).to_csv(null_info)
    #print('\n Unique Impact (Class):\n', df.hgmd_class.unique(), file=open("./data/processed/stats1.csv", "a"))
    df['hgmd_class'] = df['hgmd_class'].str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
    df['hgmd_class'] = df['hgmd_class'].str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
    df['hgmd_class'] = df['hgmd_class'].str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
    df['hgmd_class'] = df['hgmd_class'].str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
    df.drop_duplicates()
    df.dropna(axis=1, how='all', inplace=True)
    y = df['hgmd_class']
    class_dummies = pd.get_dummies(df['hgmd_class'])
    #del class_dummies[class_dummies.columns[-1]]
    print('\nImpact (Class):\n', y.value_counts(), file=open(stats, "a"))
    #y = df.hgmd_class
    df = df.drop('hgmd_class', axis=1)
    df = fill_na(df,config_dict,column_info, stats, list_tag)
    
    if 'train' in stats:
        var = df[config_dict['ML_VAR']]
        df = df.drop(config_dict['ML_VAR'], axis=1)
        df = pd.concat([class_dummies.reset_index(drop=True), df], axis=1)
        fig= plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), fmt='.2g',cmap= 'coolwarm')
        plt.savefig(f"train_{list_tag[0]}/correlation_after_{list_tag[0]}.pdf", format='pdf', dpi=1000, bbox_inches='tight')
        df = pd.concat([var, df], axis=1)
        df = df.drop(['high_impact','low_impact'], axis=1)
    return df,y
=======
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        )

        # Find features with correlation greater than 0.90
        to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
        print(
            f"Correlated columns being dropped: {to_drop}", file=open(column_info, "a")
        )

        # Drop features
        df.drop(to_drop, axis=1, inplace=True)
        df = df.reset_index(drop=True)
    print(df.columns.values.tolist(), file=open(column_info, "a"))
    # df.dropna(axis=1, how='all', inplace=True)
    df["ID"] = [f"var_{num}" for num in range(len(df))]
    print("NAs filled!")
    df = pd.concat([var.reset_index(drop=True), df], axis=1)
    return df


def main(df, config_f, stats, column_info, null_info, list_tag):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print("Config file loaded!")
    # read clinvar data

    print("\nData shape (Before filtering) =", df.shape, file=open(stats, "a"))
    df = extract_col(config_dict, df, stats, list_tag)
    print("Columns extracted! Extracting class info....")
    df.isnull().sum(axis=0).to_csv(null_info)
    # print('\n Unique Impact (Class):\n', df.hgmd_class.unique(), file=open("./data/processed/stats1.csv", "a"))
    df["hgmd_class"] = (
        df["hgmd_class"]
        .str.replace(r"DFP", "high_impact")
        .str.replace(r"DM\?", "high_impact")
        .str.replace(r"DM", "high_impact")
    )
    df["hgmd_class"] = (
        df["hgmd_class"]
        .str.replace(r"Pathogenic/Likely_pathogenic", "high_impact")
        .str.replace(r"Likely_pathogenic", "high_impact")
        .str.replace(r"Pathogenic", "high_impact")
    )
    df["hgmd_class"] = (
        df["hgmd_class"]
        .str.replace(r"DP", "low_impact")
        .str.replace(r"FP", "low_impact")
    )
    df["hgmd_class"] = (
        df["hgmd_class"]
        .str.replace(r"Benign/Likely_benign", "low_impact")
        .str.replace(r"Likely_benign", "low_impact")
        .str.replace(r"Benign", "low_impact")
    )
    df.drop_duplicates()
    df.dropna(axis=1, how="all", inplace=True)
    y = df["hgmd_class"]
    class_dummies = pd.get_dummies(df["hgmd_class"])
    # del class_dummies[class_dummies.columns[-1]]
    print("\nImpact (Class):\n", y.value_counts(), file=open(stats, "a"))
    # y = df.hgmd_class
    df = df.drop("hgmd_class", axis=1)
    df = fill_na(df, config_dict, column_info, stats, list_tag)

    if "train" in stats:
        var = df[config_dict["ML_VAR"]]
        df = df.drop(config_dict["ML_VAR"], axis=1)
        df = pd.concat([class_dummies.reset_index(drop=True), df], axis=1)
        fig = plt.figure(figsize=(20, 15))
        sns.heatmap(df.corr(), fmt=".2g", cmap="coolwarm")
        plt.savefig(
            f"train_{list_tag[0]}/correlation_after_{list_tag[0]}.pdf",
            format="pdf",
            dpi=1000,
            bbox_inches="tight",
        )
        df = pd.concat([var, df], axis=1)
        df = df.drop(["high_impact", "low_impact"], axis=1)
    return df, y

>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        required=True,
<<<<<<< HEAD
        default='nssnv',
        help="The tag used when generating train/test data. Default:'nssnv'")
=======
        default="nssnv",
        help="The tag used when generating train/test data. Default:'nssnv'",
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
<<<<<<< HEAD
        help=f"Cutoff to include at least __% of data for all rows. Default:0.5 (i.e. 50%)")
=======
        help=f"Cutoff to include at least __% of data for all rows. Default:0.5 (i.e. 50%)",
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    parser.add_argument(
        "--af-columns",
        "-af",
        type=int,
        default=0,
<<<<<<< HEAD
        help=f"To include columns with Allele frequencies or not. Default:0")
=======
        help=f"To include columns with Allele frequencies or not. Default:0",
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    parser.add_argument(
        "--af-values",
        "-afv",
        type=float,
        default=0,
<<<<<<< HEAD
        help=f"value to impute nulls for allele frequency columns. Default:0")
=======
        help=f"value to impute nulls for allele frequency columns. Default:0",
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    parser.add_argument(
        "--other-values",
        "-otv",
        type=int,
        default=0,
<<<<<<< HEAD
        help=f"Impute other columns with either custom defined values (0) or median (1). Default:0")

    args = parser.parse_args()
    list_tag = [args.var_tag, args.cutoff, args.af_columns, args.af_values, args.other_values]
    print(list_tag)
    var = list_tag[0]

    if not os.path.exists('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test'):
            os.makedirs('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test')
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test')
    print('Loading data...')
    var_f = pd.read_csv("../../interim/merged_sig_norm_class_vep-annotated.tsv", sep='\t')
    print('Data Loaded !....')
    config_f = "../../../configs/columns_config.yaml"

    #variants = ['train_non_snv','train_snv','train_snv_protein_coding','test_snv','test_non_snv','test_snv_protein_coding']
    variants = ['train_'+var, 'test_'+var]
    #variants = ['train_'+var]
    for var in variants:
        if not os.path.exists(var):
            os.makedirs(var)
        stats = var+"/stats_"+var+".csv"
        print("Filtering "+var+" variants with at-least "+str(list_tag[1]*100)+" percent data for each variant...", file=open(stats, "w"))
        #print("Filtering "+var+" variants with at-least 50 percent data for each variant...")
        column_info = var+"/"+var+'_columns.csv'
        null_info = var+"/Nulls_"+var+'.csv'
        df,y = main(var_f, config_f, stats, column_info, null_info, list_tag)
        if 'train' in stats:
=======
        help=f"Impute other columns with either custom defined values (0) or median (1). Default:0",
    )

    args = parser.parse_args()
    list_tag = [
        args.var_tag,
        args.cutoff,
        args.af_columns,
        args.af_values,
        args.other_values,
    ]
    print(list_tag)
    var = list_tag[0]

    if not os.path.exists(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test"
    ):
        os.makedirs(
            "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test"
        )
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test"
    )
    print("Loading data...")
    var_f = pd.read_csv(
        "../../interim/merged_sig_norm_class_vep-annotated.tsv", sep="\t"
    )
    print("Data Loaded !....")
    config_f = "../../../configs/columns_config.yaml"

    # variants = ['train_non_snv','train_snv','train_snv_protein_coding','test_snv','test_non_snv','test_snv_protein_coding']
    variants = ["train_" + var, "test_" + var]
    # variants = ['train_'+var]
    for var in variants:
        if not os.path.exists(var):
            os.makedirs(var)
        stats = var + "/stats_" + var + ".csv"
        print(
            "Filtering "
            + var
            + " variants with at-least "
            + str(list_tag[1] * 100)
            + " percent data for each variant...",
            file=open(stats, "w"),
        )
        # print("Filtering "+var+" variants with at-least 50 percent data for each variant...")
        column_info = var + "/" + var + "_columns.csv"
        null_info = var + "/Nulls_" + var + ".csv"
        df, y = main(var_f, config_f, stats, column_info, null_info, list_tag)
        if "train" in stats:
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
            train_columns = df.columns.values.tolist()
        else:
            df1 = pd.DataFrame()
            for key in tqdm(train_columns):
                if key in df.columns:
                    df1[key] = df[key]
                else:
                    df1[key] = 0
            df = df1
            del df1
<<<<<<< HEAD
        
        print('\nData shape (After filtering) =', df.shape, file=open(stats, "a"))
        print('Class shape=', y.shape,file=open(stats, "a"))
        print('writing to csv...')
        df.to_csv(var+'/'+'merged_data-'+var+'.csv', index=False)
        y.to_csv(var+'/'+'merged_data-y-'+var+'.csv', index=False)
        del df,y

=======

        print("\nData shape (After filtering) =", df.shape, file=open(stats, "a"))
        print("Class shape=", y.shape, file=open(stats, "a"))
        print("writing to csv...")
        df.to_csv(var + "/" + "merged_data-" + var + ".csv", index=False)
        y.to_csv(var + "/" + "merged_data-y-" + var + ".csv", index=False)
        del df, y
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
