#from numpy import mean
import numpy as np
import pandas as pd
import time
import ray
import pickle
import shap
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
import matplotlib.pyplot as plt
import yaml
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

# Start Ray.
ray.init(ignore_reinit_error=True)

output = "/models/ML_results_snp.csv"

with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

#Load data
print('\nUsing merged_data-snp.csv..', file=open(output, "a"))
X = pd.read_csv('merged_data-snp.csv')
var = df[config_dict['snps']]
X = X.drop(config_dict['snps'], axis=1)
feature_names = X.columns.tolist()
X = X.values
# X[1]
# var
y = pd.read_csv('merged_data-y-snp.csv')
#Y = pd.get_dummies(y)
Y = label_binarize(y.values, classes=['low_impact', 'high_impact']) #'Benign', 'Likely_benign', 'Uncertain_significance', 'Likely_pathogenic', 'Pathogenic'

 
#scaler = StandardScaler().fit(X)
#x_scaled = scaler.transform(X)
#scaler = MinMaxScaler(feature_range=(-1,1))
#x_scaled = scaler.fit_transform(X)
#pd.DataFrame(x_scaled, columns= columns).to_csv('scaled_data.csv', index=False)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state=42)
#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

# explain all the predictions in the test set
background = shap.kmeans(X_train, 6)

#Classifiers I wish to use
classifiers = [
	#KNeighborsClassifier(),
	#SVC(class_weight='balanced', probability=True),
    DecisionTreeClassifier(class_weight='balanced'),
    LogisticRegression(class_weight='balanced'),
    RandomForestClassifier(class_weight='balanced'),
    AdaBoostClassifier(),
    #GradientBoostingClassifier(),
	#BaggingClassifier(),
	ExtraTreesClassifier(class_weight='balanced_subsample'),
    BalancedRandomForestClassifier(),
    EasyEnsembleClassifier() # doctest: +SKIP
]

@ray.remote
def classifier(clf, X_train, X_test, Y_train, Y_test,background,feature_names):
   start = time.perf_counter()
   score = cross_validate(clf, X_train, Y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
   clf = score['estimator'][np.argmax(score['test_score'])]
   #y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
   #class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
   #clf.fit(X_train, Y_train) #, class_weight=class_weights)
   y_score = clf.predict(X_test)
   prc = precision_score(Y_test,y_score, average="weighted")
   recall  = recall_score(Y_test,y_score, average="weighted")
   roc_auc = roc_auc_score(Y_test, y_score)
   #roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
   accuracy = accuracy_score(Y_test, y_score)
   #score = clf.score(X_train, Y_train)
   #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
   matrix = confusion_matrix(Y_test, y_score,)
   finish = (time.perf_counter()-start)/60
   clf_name = str(type(clf)).split("'")[1].split(".")[3]

   # explain all the predictions in the test set
   background = shap.kmeans(X_train, 6)
   explainer = shap.KernelExplainer(clf.predict, background)
   background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
   shap_values = explainer.shap_values(background)
   plt.figure()
   shap.summary_plot(shap_values, background, feature_names, show=False)
   #shap.plots.waterfall(shap_values[0], max_display=15)
   plt.savefig("./models/"+clf_name + "_snp_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')

   #list1 = [clf_name ,prc, roc_auc, accuracy, score, matrix, finish]
   list1 = [clf_name, np.mean(score['train_score']), np.mean(score['test_score']), prc, recall, roc_auc, accuracy, finish, matrix]
   pickle.dump(clf, open("./models/"+clf_name+"_snp.pkl", 'wb'))
   return list1

print('Model\tTrain_score(train_data)\tTest_score(train_data)\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
for i in classifiers:
   list1 = ray.get(classifier.remote(i, X_train, X_test, Y_train, Y_test,background,feature_names))
   print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\t{list1[7]}\n{list1[8]}', file=open(output, "a"))


print('done!')