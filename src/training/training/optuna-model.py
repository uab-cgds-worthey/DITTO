#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 23:42:18 2020

@author: tarunmamidi
"""

import pandas as pd
import numpy as np; np.random.seed(5)  
import tensorflow as tf
import tensorflow.keras as keras
try:
    tf.get_logger().setLevel('INFO')
except Exception as exc:
    print(exc)
import warnings
warnings.simplefilter("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import shap
shap.initjs()
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

#n_columns = 112

print("Loading data....\n")

x = pd.read_csv('clinvar-md.csv')
var = x[['AAChange.refGene','ID']]
X = x.drop(['AAChange.refGene','ID'], axis=1)
features = X.columns.tolist()
#var.to_csv('/Users/tarunmamidi/Documents/Development/ditto/tune/variant_ID.csv')
X=X.values
y = pd.read_csv('clinvar-y-md.csv')
Y = pd.get_dummies(y)
#Y = label_binarize(y, classes=['Benign', 'Pathogenic'])
train_x,test_x, train_y, test_y= train_test_split(X,Y,test_size=.30,random_state=42)
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
    #n_columns = train_x.shape[1]
    #n_columns = 141
print('Data loaded and scaled.\n')

#4-class
#parameters = {'n_layers': 2, 'activation': 'sigmoid', 'n_units_l0': 42, 'kernel_initializer_l0': 'he_normal', 'activation_l0': 'sigmoid', 'dropout_l0': 0.11794277735529887, 'n_units_l1': 14, 'kernel_initializer_l1': 'zero', 'activation_l1': 'softmax', 'dropout_l1': 0.44728594031480523, 'kernel_initializer': 'he_normal', 'optimizer': 'Adam', 'batch_size': 658}
#6-class - LR
#parameters = {'n_layers': 5, 'activation': 'relu', 'n_units_l0': 168, 'kernel_initializer_l0': 'glorot_uniform', 'activation_l0': 'relu', 'dropout_l0': 0.12849446365413872, 'n_units_l1': 299, 'kernel_initializer_l1': 'uniform', 'activation_l1': 'elu', 'dropout_l1': 0.2952461200802723, 'n_units_l2': 280, 'kernel_initializer_l2': 'normal', 'activation_l2': 'softplus', 'dropout_l2': 0.8957821599147904, 'n_units_l3': 14, 'kernel_initializer_l3': 'normal', 'activation_l3': 'linear', 'dropout_l3': 0.13794225422380274, 'n_units_l4': 169, 'kernel_initializer_l4': 'normal', 'activation_l4': 'softsign', 'dropout_l4': 0.6453839014735023, 'kernel_initializer': 'uniform', 'optimizer': 'Adamax', 'batch_size': 610}
#6-class - median
parameters = {'n_layers': 2, 'activation': 'elu', 'n_units_l0': 215, 'kernel_initializer_l0': 'glorot_normal', 'activation_l0': 'hard_sigmoid', 'dropout_l0': 0.7248274834825591, 'n_units_l1': 35, 'kernel_initializer_l1': 'lecun_uniform', 'activation_l1': 'elu', 'dropout_l1': 0.4279528310227717, 'kernel_initializer': 'normal', 'optimizer': 'Adamax', 'batch_size': 497}

def tune_data(config): 
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(Dense(X.shape[1], input_shape=(X.shape[1],), activation=config['activation']))
    for i in range(config['n_layers']):
        model.add(Dense(config['n_units_l{}'.format(i)], name = "dense_l{}".format(i), kernel_initializer=config["kernel_initializer_l{}".format(i)], activation = config["activation_l{}".format(i)]))
        model.add(Dropout( config["dropout_l{}".format(i)]))
    model.add(Dense(units = Y.shape[1], name = "dense_last", kernel_initializer=config["kernel_initializer"],  activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=config["optimizer"], metrics=['accuracy'])
    model.summary()
    # Train the model
    model.fit(
        train_x, train_y, 
        verbose=2,
        batch_size=config['batch_size'], 
        epochs=150)
    # Evaluate the model accuracy on the validation set.
    #score = model.evaluate(test_x, test_y, verbose=0)
    return model


model = tune_data(parameters)
results = model.evaluate(test_x, test_y)
y_score = model.predict(test_x)
prc = average_precision_score(test_y, y_score, average=None)
prc_micro = average_precision_score(test_y, y_score, average='micro')
#matrix = confusion_matrix(np.argmax(test_y, axis=1), np.argmax(y_score, axis=1))
matrix = confusion_matrix(np.argmax(test_y.values,axis=1), np.argmax(y_score, axis=1))
print(f'Test loss: {results[0]}\nTest accuracy: {results[1]}\nOverall precision score: {prc_micro}\nPrecision score: {prc}\nConfusion matrix:\n{matrix}') #,, file=open("Ditto-v0.csv", "a")

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
model.save("my_model")
model.save_weights("weights.h5")

#from tensorflow import keras
#my_model = keras.models.load_model('my_model')
#my_model.load_weights("weights.h5")

#test_x = scaler.transform(X)
#y_score = model.predict(test_x)
#pred = pd.DataFrame(y_score, columns = ['pred_Benign','pred_VUS', 'pred_Pathogenic'])
#var = var.to_frame()
#classified = pd.concat([var.reset_index(drop=True), pred], axis=1)
#overall = classified.merge(df1,on='CLNHGVS')
#overall.to_csv('predicted_results.csv', index=False)
test_y = test_y.sort_index(ascending=True)
mis = np.where(np.argmax(test_y.values,axis=1) != np.argmax(y_score, axis=1))[0].tolist()
var = var.loc[var.index.isin(test_y.index)]
var = pd.concat([var, test_y], axis=1)

pred = pd.DataFrame(y_score, columns = ['pred_Benign','pred_Pathogenic'])
var = pd.concat([var.reset_index(drop=True), pred], axis=1)
var = var.loc[var.index.isin(mis)]
#var = pd.concat([var.reset_index(drop=True), x], axis=1)
misclass = var.merge(x, on='ID')

#true = test_y
#pred = pred.loc[pred.index.isin(mis)]
#true = true.loc[true.index.isin(mis)]
#misclass = var.merge(true, how='outer', left_index=True, right_index=True).merge(pred, how='outer', left_index=True, right_index=True)
misclass.to_csv('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/misclassified_pb-6.csv', index = False)

x = scaler.fit_transform(X)
background = shap.kmeans(train_x, 6)
explainer = shap.KernelExplainer(model.predict, background)
print("base value =", explainer.expected_value)
background = x[np.random.choice(x.shape[0], 1000, replace=False)]
shap_values = explainer.shap_values(background) #, nsamples=500
shap.summary_plot(shap_values, x, features, show=False)

import matplotlib.pyplot as pl
pl.savefig("summary_plot.pdf", format='pdf', dpi=1000, bbox_inches='tight')