#from numpy import mean
import numpy as np
import pandas as pd
import time
import ray
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
#from skopt import BayesSearchCV
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
#from sklearn.metrics import precision_score

import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

# Start Ray.
ray.init(ignore_reinit_error=True)



#Load data
print('\nUsing hgmd-md.csv..\n', file=open("ML_results.csv", "a"))
X = pd.read_csv('hgmd-md.csv')
var = X[['SYMBOL','Feature','Consequence','ID']]
X = X.drop(['SYMBOL','Feature','Consequence', 'ID'], axis=1)
X = X.values
# X[1]
# var
y = pd.read_csv('hgmd-y-md.csv')
#Y = pd.get_dummies(y)
Y = label_binarize(y.values, classes=['low_impact', 'high_impact']) #'Benign', 'Likely_benign', 'Uncertain_significance', 'Likely_pathogenic', 'Pathogenic'

 
#scaler = StandardScaler().fit(X)
#x_scaled = scaler.transform(X)
#scaler = MinMaxScaler(feature_range=(-1,1))
#x_scaled = scaler.fit_transform(X)
#pd.DataFrame(x_scaled, columns= columns).to_csv('scaled_data.csv', index=False)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Classifiers I wish to use
classifiers = [
	#KNeighborsClassifier(),
	#SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
	#BaggingClassifier(),
	ExtraTreesClassifier(),
    BalancedRandomForestClassifier()
    #EasyEnsembleClassifier() # doctest: +SKIP
]

# @ray.remote
# def classifier(model, X_train, X_test, column):
#     #print(classifier, file=open("all-model-scores.csv", "a"))
#     model = RFE(model)
#     model.fit(X_train, Y_train)
#     df = pd.DataFrame(model.ranking_,index=column,columns=['Rank']).sort_values(by='Rank',ascending=True)
#     df.to_csv('ranking.csv', mode='a')
#     finish = time.perf_counter()
#     list1 = [model, finish]
#     return list1

# for i in classifiers:
#     list1 = ray.get(classifier.remote(i, X_train, X_test, columns))
#     time=(list1[1]-start)/60
#     print(f'{list1[0]}\t{time}')


@ray.remote
def classifier(clf, X_train, X_test, Y_train, Y_test):
   #print(classifier, file=open("all-model-scores.csv", "a"))
   #clf = OneVsRestClassifier(model)
   start = time.perf_counter()
   clf.fit(X_train, Y_train)
   y_score = clf.predict_proba(X_test)
   prc = average_precision_score(Y_test, np.argmax(y_score, axis=1), average=None)
   prc_micro = average_precision_score(Y_test, np.argmax(y_score, axis=1), average='micro')
   score = clf.score(X_train, Y_train)
   #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
   matrix = confusion_matrix(Y_test, np.argmax(y_score, axis=1))
   finish = (time.perf_counter()-start)/60
   list1 = [clf ,prc, prc_micro, score, matrix, finish]
   return list1

print('Model\tprecision_score\taverage_precision_score\tTrain_score\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open("ML_results.csv", "a"))
for i in classifiers:
   list1 = ray.get(classifier.remote(i, X_train, X_test, Y_train, Y_test))
   #list1 = classifier(i, X_train, X_test, Y_train, Y_test)
   
   print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[5]}\n{list1[4]}', file=open("ML_results.csv", "a"))

print('done!')
#clf = OneVsRestClassifier(DecisionTreeClassifier())
#clf.fit(X_train, Y_train)
#y_score = clf.predict_proba(X_test)
#confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
#average_precision_score(Y_test, y_score, average='micro')
#average_precision_score(Y_test, y_score, average=None)