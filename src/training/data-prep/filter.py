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
    if 'train' in stats:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
    else:
        df= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_test'])]
    
    if 'train' in stats:
        print('Dropping empty columns and rows along with duplicate rows...')
        #df.dropna(axis=1, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[0]/4)
        df.dropna(axis=0, thresh=(df.shape[1]*0.3), inplace=True)  #thresh=(df.shape[1]*0.3),   how='all',
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

def fill_na(df,config_dict, column_info, stats): #(config_dict,df):
    #df1  = pd.DataFrame()
    var = df[config_dict['var']]
    df = df.drop(config_dict['var'], axis=1)
    print('parsing difficult columns......')
    #if 'non_snv' in stats:
    df['GERP'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['GERP'].str.split('&')]
    if 'nssnv' in stats:
        df['MutationTaster_score'] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df['MutationTaster_score'].str.split('&')]
    #else:
    #    for col in tqdm(config_dict['col_conv']):
    #        df[col] = [np.mean([float(item.replace('.', '0')) if item == '.' else float(item) for item in i]) if type(i) is list else i for i in df[col].str.split('&')]
    print('One-hot encoding...')
    df = pd.get_dummies(df, prefix_sep='_')
    print(df.columns.values.tolist(),file=open(column_info, "w"))
    #df.head(2).to_csv(column_info, index=False)
    #lr = LinearRegression()
    #imp= IterativeImputer(estimator=lr, verbose=2, max_iter=10, tol=1e-10, imputation_order='roman')
    print('Filling NAs ....')
    #df = imp.fit_transform(df)
    #df = pd.DataFrame(df, columns = columns)
    #df=df.fillna(df.median())
    df1 = pd.DataFrame()
    if 'non_snv' in stats:
        for key in tqdm(config_dict['non_snv_columns']):
            if key in df.columns:
                df1[key] = df[key].fillna(config_dict['non_snv_columns'][key]).astype('float64')
            else:
                df1[key] = config_dict['non_snv_columns'][key]
    else:
        for key in tqdm(config_dict['snv_columns']):
            if key in df.columns:
                df1[key] = df[key].fillna(config_dict['snv_columns'][key]).astype('float64')
            else:
                df1[key] = config_dict['snv_columns'][key]
    df = df1
    del df1
    df = df.reset_index(drop=True)
    df['ID'] = [f'var_{num}' for num in range(len(df))]
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
    print('Columns extracted! Extracting class info....')
    df.isnull().sum(axis = 0).to_csv(null_info)
    #print('\n Unique Impact (Class):\n', df.hgmd_class.unique(), file=open("./data/processed/stats1.csv", "a"))
    y = df.hgmd_class.str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
    y = y.str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
    y = y.str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
    y = y.str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
    print('\nImpact (Class):\n', y.value_counts(), file=open(stats, "a"))
    #y = df.hgmd_class
    df = df.drop('hgmd_class', axis=1)
    df = fill_na(df,config_dict,column_info, stats) #(config_dict,df)
    print('\nData shape (After filtering) =', df.shape, file=open(stats, "a"))
    print('Class shape=', y.shape,file=open(stats, "a"))
    return df,y

if __name__ == "__main__":
    os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    print('Loading data...')
    var_f = pd.read_csv("../interim/merged_sig_norm_class_vep-annotated.tsv", sep='\t')
    print('Data Loaded !....')
    config_f = "../../configs/columns_config.yaml"

    #variants = ['train_non_snv','train_snv','train_snv_protein_coding','test_snv','test_non_snv','test_snv_protein_coding']
    variants = ['train_nssnv', 'test_nssnv']
    for var in variants:
        if not os.path.exists(var):
            os.mkdir(var)
        stats = var+"/stats_"+var+".csv"
        print("Filtering "+var+" variants with at-least 30 percent data for each variant...", file=open(stats, "w"))
        print("Filtering "+var+" variants with at-least 30 percent data for each variant...")
        column_info = var+"/"+var+'_columns.csv'
        null_info = var+"/Nulls_"+var+'.csv'
        df,y = main(var_f, config_f, stats, column_info, null_info)
        print('writing to csv...')
        df.to_csv(var+'/'+'merged_data-'+var+'.csv', index=False)
        y.to_csv(var+'/'+'merged_data-y-'+var+'.csv', index=False)
        del df,y

