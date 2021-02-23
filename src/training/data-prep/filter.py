#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
pd.set_option('display.max_rows', None)
import numpy as np
from tqdm import tqdm 
import yaml
import os
#from sklearn.linear_model import LinearRegression
#from sklearn.experimental import enable_iterative_imputer
#from sklearn.impute import IterativeImputer
#import pickle

def get_col_configs(config_f):
    with open(config_f) as fh:
        config_dict = yaml.safe_load(fh)

    # print(config_dict)
    return config_dict

def extract_col(config_dict,df, stats):
    print('Extracting columns and rows according to config file !....')
    df = df[config_dict['columns']]
    df= df.loc[df['hgmd_class'].isin(config_dict['ClinicalSignificance'])]
    print('Dropping empty columns and rows...')
    #df.replace('.', np.nan, inplace=True)
    #df[df.isnull().sum(axis=1)]
    df.dropna(axis=1, how='all', inplace=True)  #thresh=(df.shape[0]/4)
    df.dropna(axis=0, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[1]/1.2)
    print('\nhgmd_class:\n', df['hgmd_class'].value_counts(), file=open(stats, "a"))
    print('\nclinvar_CLNSIG:\n', df['clinvar_CLNSIG'].value_counts(), file=open(stats, "a"))
    print('\nclinvar_CLNREVSTAT:\n', df['clinvar_CLNREVSTAT'].value_counts(), file=open(stats, "a"))
    print('\nConsequence:\n', df['Consequence'].value_counts(), file=open(stats, "a"))
    print('\nIMPACT:\n', df['IMPACT'].value_counts(), file=open(stats, "a"))
    print('\nBIOTYPE:\n', df['BIOTYPE'].value_counts(), file=open(stats, "a"))
    #df = df.drop(['CLNVC','MC'], axis=1)
    # CLNREVSTAT, CLNVC, MC
    return df

def fill_na(df): #(config_dict,df):
    #df1  = pd.DataFrame()
    var = df[['SYMBOL','Feature','Consequence']]
    df = df.drop(['SYMBOL','Feature','Consequence','clinvar_CLNREVSTAT','clinvar_CLNSIG'], axis=1)
    df.dtypes.to_csv('./data/processed/columns1.csv')
    #df.to_csv('./data/interim/temp.csv', index=False)
    #df = pd.read_csv('./data/interim/temp.csv')
    #os.remove('./data/interim/temp.csv')
    print('One-hot encoding...')
    df = pd.get_dummies(df, prefix_sep='_')
    df.head(2).to_csv('./data/processed/merged_data_columns.csv', index=False)
    #lr = LinearRegression()
    #imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print('Filling NAs ....')
    #df = imp.fit_transform(df)
    #filehandler = open('./data/processed/imputer.pkl', 'wb') 
    #pickle.dump(imp, filehandler)
    #df = pd.DataFrame(df, columns = columns)
    df=df.fillna(0) #(df.median())
    df = df.reset_index(drop=True)
    df['ID'] = [f'var_{num}' for num in range(len(df))]
    #for key in tqdm(config_dict['Fill_NAs']):
    #    if key in df.columns:
    #        df1[key] = df[key].fillna(config_dict['Fill_NAs'][key]).astype('float64')
    #    else:
    #        df1[key] = config_dict['Fill_NAs'][key]
    print('NAs filled!')
    df = pd.concat([var.reset_index(drop=True), df], axis=1)
    return df

def main(var_f, config_f, stats):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print('Config file loaded!\nNow loading data.....\n')
    # read clinvar data
    df = pd.read_csv(var_f, sep='\t')
    print('Data Loaded !....')
    print('\nData shape (Before filtering) =', df.shape, file=open(stats, "a"))
    df = extract_col(config_dict,df, stats)
    print('Columns extracted !....')
    df.isnull().sum(axis = 0).to_csv('./data/processed/NA-counts.csv')
    #print('\n Unique Impact (Class):\n', df.hgmd_class.unique(), file=open("./data/processed/stats1.csv", "a"))
    y = df.hgmd_class.str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
    y = y.str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
    y = y.str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
    y = y.str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
    print('\nImpact (Class):\n', y.value_counts(), file=open(stats, "a"))
    #y = df.hgmd_class
    df = df.drop('hgmd_class', axis=1)
    df = fill_na(df) #(config_dict,df)
    #print dataframe shape
    #df.dtypes.to_csv('../../data/interim/head.csv')
    print('\nData shape (After filtering) =', df.shape, file=open(stats, "a"))
    print('Class shape=', y.shape,file=open(stats, "a"))
    df.to_csv('./data/processed/merged_data-null.csv', index=False)
    y.to_csv('./data/processed/merged_data-y-null.csv', index=False)
    return None

if __name__ == "__main__":
    os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/')
    var_f = "./data/processed/merged_sig_norm_class_vep-annotated.tsv"
    config_f = "./configs/columns_config.yaml"
    stats = "./data/processed/stats1.csv"
    main(var_f, config_f, stats)
