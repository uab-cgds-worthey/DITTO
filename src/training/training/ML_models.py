#from numpy import mean
import numpy as np
import pandas as pd
import time
import ray
import pickle
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

# Start Ray.
ray.init(ignore_reinit_error=True)



#Load data
print('\nUsing merged_data-md.csv..\n', file=open("ML_results.csv", "a"))
X = pd.read_csv('merged_data-md.csv')
var = X[['SYMBOL','Feature','Consequence','ID']]
X = X.drop(['SYMBOL','Feature','Consequence', 'ID'], axis=1)
X = X.values
# X[1]
# var
y = pd.read_csv('merged_data-y-md.csv')
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
	#SVC(class_weight='balanced', probability=True),
    #DecisionTreeClassifier(class_weight='balanced'),
    #LogisticRegression(class_weight='balanced'),
    RandomForestClassifier(class_weight='balanced')
    #AdaBoostClassifier(),
    ##GradientBoostingClassifier(),
	##BaggingClassifier(),
	#ExtraTreesClassifier(class_weight='balanced_subsample'),
    #BalancedRandomForestClassifier(),
    #EasyEnsembleClassifier() # doctest: +SKIP
]

@ray.remote
def classifier(clf, X_train, X_test, Y_train, Y_test):
   start = time.perf_counter()
   score = cross_validate(clf, X_train, Y_train, cv=5, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
   clf = score['estimator'][np.argmax(score['test_score'])]
   y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
   #class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
   #clf.fit(X_train, Y_train) #, class_weight=class_weights)
   #y_score = clf.predict_proba(X_test)
   #prc = average_precision_score(Y_test, np.argmax(y_score, axis=1), average=None)
   #roc_auc = roc_auc_score(Y_test, y_score)
   #roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
   #accuracy = accuracy_score(Y_test, np.argmax(y_score, axis=1))
   #score = clf.score(X_train, Y_train)
   #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
   #matrix = confusion_matrix(Y_test, np.argmax(y_score, axis=1))
   finish = (time.perf_counter()-start)/60
   clf_name = str(type(clf)).split("'")[1].split(".")[3]

   #list1 = [clf_name ,prc, roc_auc, accuracy, score, matrix, finish]
   list1 = [clf_name, np.mean(score['train_score']), np.mean(score['test_score']),  finish, y_score]
   pickle.dump(clf, open("./models/"+clf_name+".pkl", 'wb'))
   return list1

print('Model\tTrain_score\tTest_score\tTime(min)', file=open("ML_results.csv", "a"))    #\tConfusion_matrix[low_impact, high_impact]
for i in classifiers:
   list1 = ray.get(classifier.remote(i, X_train, X_test, Y_train, Y_test))
   print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}', file=open("ML_results.csv", "a"))
   #print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[6]}\n{list1[5]}', file=open("ML_results.csv", "a"))


print('done!')