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
    #df = df[config_dict['Consequence']]
    #df= df.loc[df['Consequence'].isin(config_dict['Consequence'])]
    if 'non-snv' in stats:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clin_non_snv'])]
        df=df[(df['Alternate Allele'].str.len() > 1) | (df['Reference Allele'].str.len() > 1)]
        print('\nData shape (non-snv) =', df.shape, file=open(stats, "a"))
        #print('Dropping empty columns and rows...')
        #df.dropna(axis=1, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[0]/4)
        #df.dropna(axis=0, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[1]*0.3),   how='all',
        #df.dropna(axis=1, how='all', inplace=True)  #thresh=(df.shape[0]/4)
    else:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clin_snv'])]
        df=df[(df['Alternate Allele'].str.len() < 2) & (df['Reference Allele'].str.len() < 2)]
        #df = df[df['BIOTYPE']=='protein_coding']
        print('\nData shape (snv) =', df.shape, file=open(stats, "a"))
    print('Dropping empty columns and rows...')
    #df.dropna(axis=1, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[0]/4)
    df.dropna(axis=0, thresh=(df.shape[1]*0.5), inplace=True)  #thresh=(df.shape[1]*0.3),   how='all',
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

def fill_na(df,config_dict, column_info): #(config_dict,df):
    #df1  = pd.DataFrame()
    var = df[config_dict['var']]
    df = df.drop(config_dict['var'], axis=1)
    #df.dtypes.to_csv('./data/processed/columns1.csv')
    #df.to_csv('./data/interim/temp.csv', index=False)
    #df = pd.read_csv('./data/interim/temp.csv')
    #os.remove('./data/interim/temp.csv')
    print('One-hot encoding...')
    df = pd.get_dummies(df, prefix_sep='_')
    print(df.columns.values.tolist(),file=open(column_info, "w"))
    #df.head(2).to_csv(column_info, index=False)
    #lr = LinearRegression()
    #imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print('Filling NAs ....')
    #df = imp.fit_transform(df)
    #filehandler = open('./data/processed/imputer.pkl', 'wb') 
    #pickle.dump(imp, filehandler)
    #df = pd.DataFrame(df, columns = columns)
    df=df.fillna(df.median())
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

def main(df, config_f, stats,column_info, null_info):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print('Config file loaded!')
    # read clinvar data
    
    print('\nData shape (Before filtering) =', df.shape, file=open(stats, "a"))
    df = extract_col(config_dict,df, stats)
    print('Columns extracted !....')
    df.isnull().sum(axis = 0).to_csv(null_info)
    #print('\n Unique Impact (Class):\n', df.hgmd_class.unique(), file=open("./data/processed/stats1.csv", "a"))
    y = df.hgmd_class.str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
    y = y.str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
    y = y.str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
    y = y.str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
    print('\nImpact (Class):\n', y.value_counts(), file=open(stats, "a"))
    #y = df.hgmd_class
    df = df.drop('hgmd_class', axis=1)
    df = fill_na(df,config_dict,column_info) #(config_dict,df)
    #print dataframe shape
    #df.dtypes.to_csv('../../data/interim/head.csv')
    print('\nData shape (After filtering) =', df.shape, file=open(stats, "a"))
    print('Class shape=', y.shape,file=open(stats, "a"))
    return df,y

if __name__ == "__main__":
    os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/')
    print('Loading data...')
    var_f = pd.read_csv("./data/processed/merged_sig_norm_class_vep-annotated.tsv", sep='\t')
    print('Data Loaded !....')
    config_f = "./configs/columns_config.yaml"

    #For non-snv variants
    #stats = "./data/processed/stats-non-snv.csv"
    #print("Filtering non-snv variants with at-least 50 percent data for each variant...", file=open(stats, "a"))
    #print("Filtering non-snv variants with at-least 50 percent data for each variant...")
    #column_info = './data/processed/non_snv_columns.csv'
    #null_info = './data/processed/NA-counts-non-snv.csv'
    #df,y = main(var_f, config_f, stats, column_info, null_info)
    #df.to_csv('./data/processed/merged_data-non-snv.csv', index=False)
    #y.to_csv('./data/processed/merged_data-y-non-snv.csv', index=False)
    #del df,y

    #For snv variants
    stats = "./data/processed/stats-snv.csv"
    print("Filtering snv variants with at-least 50 percent data for each variant...", file=open(stats, "a"))
    print("Filtering snv variants with at-least 50 percent data for each variant...")
    column_info = './data/processed/snv_columns.csv'
    null_info = './data/processed/NA-counts-snv.csv'
    df,y = main(var_f, config_f, stats, column_info, null_info)
    df.to_csv('./data/processed/merged_data-snv.csv', index=False)
    y.to_csv('./data/processed/merged_data-y-snv.csv', index=False)
