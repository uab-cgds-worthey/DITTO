#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import os
import yaml
import pickle
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

print("Loading data....")

with open('SL212589_genes.yaml') as fh:
        config_dict = yaml.safe_load(fh)

X = pd.read_csv('sample1-filtered.csv')
print('Data Loaded!')
#overall.loc[:, overall.columns.str.startswith('CLN')]
var = X[['AAChange.refGene','ID']]
X = X.drop(['AAChange.refGene','ID'], axis=1)
X=X.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
print('Data scaled and Loading Ditto....!')
#y = pd.read_csv('clinvar-y-3.csv')
model = keras.models.load_model('my_model')
model.load_weights("weights.h5")
print('Ditto Loaded!\nRunning predictions.....')


#SL135596	MSTO1 	MYOPATHY, MITOCHONDRIAL, AND ATAXIA		MSTO1(NM_018116.3,c.676C>T,p.Gln226Ter,Likely Pathogenic);	MSTO1(NM_018116.3,c.971C>T,p.Thr324Ile,VUS)

y_score = model.predict(X)
print('Predictions finished!Sorting ....')
pred = pd.DataFrame(y_score, columns = ['pred_Benign', 'pred_Pathogenic'])

overall = pd.concat([var, pred], axis=1)
#classified1 = pd.concat([y.reset_index(drop=True), classified], axis=1)
del X
X= pd.read_csv("../interim/filtered_sample.csv")

overall = overall.merge(X,on='ID')
overall['hazel'] = X['Gene.refGene'].map(config_dict)
del X
overall['hazel'] = overall['hazel'].fillna(0)
overall['HD'] = (overall['pred_Pathogenic']+overall['hazel'])/2
overall.drop_duplicates(inplace=True)
overall = overall.reset_index(drop=True)
overall = overall.sort_values([ 'HD', 'pred_Pathogenic'], ascending=[False,False])
overall.head(500).to_csv('predicted_results_500_SL212589.csv', index=False)
overall = overall.sort_values([ 'CHROM', 'POS'])
#columns = overall.columns
print('writing to database...')
#overall.head(500).to_csv('predicted_results_500_SL212589.tsv', sep='\t', index=False)
overall.to_csv('predicted_results_SL212589.tsv', sep='\t', index=False)
#overall.loc[:, overall.columns.str.startswith('CLNS')]

#df1 = overall.iloc[:, :90]
#df2 = overall.iloc[:, 90:]
#df2 = pd.concat([df1['ID'], df2], axis=1)
#del overall
#store = pd.HDFStore("predicted_results_SL212589.h5")
#store.append("SL212589", overall, min_itemsize={"values": 100}, data_columns=columns)
#overall.to_hdf("predicted_results_SL212589.h5", "SL212589", format="table", mode="w")
#pd.read_hdf("store_tl.h5", "table", where=["index>2"])
#from sqlalchemy import event
#engine1 = create_engine('sqlite:///SL212589_1.db', echo=True, pool_pre_ping=True)
#engine = create_engine('sqlite:///SL212589.db', echo=True, pool_pre_ping=True)
#sqlite_connection = engine.connect()
#sqlite_connection1 = engine1.connect()
#sqlite_table = 'SL212589'
#overall.to_sql(sqlite_table, sqlite_connection, index=False, if_exists="append", chunksize=10000, method='multi') #chunksize=10000,
#
#df1.to_sql(sqlite_table, sqlite_connection, if_exists='fail')
#df2.to_sql(sqlite_table, sqlite_connection1, if_exists='fail')
#sqlite_connection.close()
print('Database storage complete!')
