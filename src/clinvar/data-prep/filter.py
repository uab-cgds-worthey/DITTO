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

def extract_col(config_dict,df):
    print('Extracting columns and rows according to config file !....')
    df = df[config_dict['columns']]
    df= df.loc[df['clinvar_CLNSIG'].isin(config_dict['ClinicalSignificance']['6-class'])]
    print('Dropping empty columns and rows...')
    #df.replace('.', np.nan, inplace=True)
    #df[df.isnull().sum(axis=1)]
    df.dropna(axis=1, how='all', inplace=True)  #thresh=(df.shape[0]/4)
    df.dropna(axis=0, how='all', inplace=True)  #thresh=(df.shape[1]/1.2)
    print('\nclinvar_CLNSIG:\n', df['clinvar_CLNSIG'].value_counts(), file=open("./data/processed/stats1.csv", "a"))
    print('\nclinvar_CLNREVSTAT:\n', df['clinvar_CLNREVSTAT'].value_counts(), file=open("./data/processed/stats1.csv", "a"))
    print('\nConsequence:\n', df['Consequence'].value_counts(), file=open("./data/processed/stats1.csv", "a"))
    print('\nIMPACT:\n', df['IMPACT'].value_counts(), file=open("./data/processed/stats1.csv", "a"))
    print('\nBIOTYPE:\n', df['BIOTYPE'].value_counts(), file=open("./data/processed/stats1.csv", "a"))
    #df = df.drop(['CLNVC','MC'], axis=1)
    # CLNREVSTAT, CLNVC, MC
    return df

def fill_na(df): #(config_dict,df):
    #df1  = pd.DataFrame()
    var = df[['SYMBOL','Feature','Consequence']]
    df = df.drop(['SYMBOL','Feature','Consequence'], axis=1)
    df.dtypes.to_csv('./data/processed/columns1.csv')
    #df.to_csv('./data/interim/temp.csv', index=False)
    #df = pd.read_csv('./data/interim/temp.csv')
    #os.remove('./data/interim/temp.csv')
    print('One-hot encoding...')
    df = pd.get_dummies(df, prefix_sep='_')
    df.head(2).to_csv('./data/processed/clinvar_columns1.csv', index=False)
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

def main(var_f, config_f):
    # read QA config file
    config_dict = get_col_configs(config_f)
    print('Config file loaded!\nNow loading data.....\n')
    # read clinvar data
    df = pd.read_csv(var_f, sep='\t')
    print('Data Loaded !....')
    df = extract_col(config_dict,df)
    print('Columns extracted !....')
    df.isnull().sum(axis = 0).to_csv('./data/processed/NA-counts-6-class.csv')
    y = df.clinvar_CLNSIG.str.replace(r'/Likely_pathogenic','').str.replace(r'/Likely_benign','')
    y = y.str.replace(r'Likely_benign','Benign').str.replace(r'Likely_pathogenic','Pathogenic')
    df = df.drop('clinvar_CLNSIG', axis=1)
    df = fill_na(df) #(config_dict,df)
    #print dataframe shape
    #df.dtypes.to_csv('../../data/interim/head.csv')
    print('\nData shape=', df.shape, file=open("./data/processed/stats1.csv", "a"))
    print('Class shape=', y.shape, file=open("./data/processed/stats1.csv", "a"))
    df.to_csv('./data/processed/clinvar1-md.csv', index=False)
    y.to_csv('./data/processed/clinvar1-y-md.csv', index=False)
    return None

if __name__ == "__main__":
    os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/')
    var_f = "./data/processed/vep/clinvar_vep-annotated.tsv"
    config_f = "./configs/columns_config.yaml"
    
    main(var_f, config_f)
