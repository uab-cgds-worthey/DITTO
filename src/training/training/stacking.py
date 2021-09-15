#from numpy import mean
import numpy as np
import pandas as pd
import time
import argparse
import ray
# Start Ray.
ray.init(ignore_reinit_error=True)
import warnings
warnings.simplefilter("ignore")
from joblib import dump, load
import shap
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import functools  
print = functools.partial(print, flush=True)
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import gc
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test')

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

#@ray.remote(num_returns=6)
def data_parsing(var,config_dict,output):
    #Load data
    #print(f'\nUsing merged_data-train_{var}..', file=open(output, "a"))
    X_train = pd.read_csv(f'train_{var}/merged_data-train_{var}.csv')
    #var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict['ML_VAR'], axis=1)
    feature_names = X_train.columns.tolist()
    X_train = X_train.values
    Y_train = pd.read_csv(f'train_{var}/merged_data-y-train_{var}.csv')
    Y_train = label_binarize(Y_train.values, classes=['low_impact', 'high_impact']).ravel() 

    X_test = pd.read_csv(f'test_{var}/merged_data-test_{var}.csv')
    #var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
    #feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f'test_{var}/merged_data-y-test_{var}.csv')
    print('Data Loaded!')
    #Y = pd.get_dummies(y)
    Y_test = label_binarize(Y_test.values, classes=['low_impact', 'high_impact']).ravel()  
    
    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    # explain all the predictions in the test set
    background = shap.kmeans(X_train, 10)
    return X_train, X_test, Y_train, Y_test, background, feature_names


#@ray.remote #(num_cpus=9)
def classifier(clf,var, X_train, X_test, Y_train, Y_test, background,feature_names, output):
   start = time.perf_counter()
   #score = cross_validate(clf, X_train, Y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0, scoring=('roc_auc','neg_log_loss'))
   #clf = score['estimator'][np.argmin(score['test_neg_log_loss'])]
   #y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
   #class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
   clf.fit(X_train, Y_train) #, class_weight=class_weights)
   clf_name = "StackingClassifier"
   with open(f"../models/{var}/{clf_name}_{var}.joblib", 'wb') as f:
    dump(clf, f, compress='lz4')
   #del clf
   #with open(f"./models/{var}/{clf_name}_{var}.joblib", 'rb') as f:
   # clf = load(f)
   train_score = clf.score(X_train, Y_train)
   y_score = clf.predict(X_test)
   prc = precision_score(Y_test,y_score, average="weighted")
   recall  = recall_score(Y_test,y_score, average="weighted")
   roc_auc = roc_auc_score(Y_test, y_score)
   #roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
   accuracy = accuracy_score(Y_test, y_score)
   #score = clf.score(X_train, Y_train)
   #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
   matrix = confusion_matrix(Y_test, y_score)
   finish = (time.perf_counter()-start)/60
   with open(output, 'a') as f:
        f.write(f"{clf_name}\t{train_score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}\n")

   # explain all the predictions in the test set
   #background = shap.kmeans(X_train, 6)
   explainer = shap.KernelExplainer(clf.predict, background)
   del clf, X_train
   background = X_test[np.random.choice(X_test.shape[0], 10000, replace=False)]
   shap_values = explainer.shap_values(background)
   plt.figure()
   shap.summary_plot(shap_values, background, feature_names, show=False)
   #shap.plots.waterfall(shap_values[0], max_display=15)
   plt.savefig(f"../models/{var}/{clf_name}_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
   return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        required=True,
        default='nssnv',
        help="The tag used when generating train/test data. Default:'nssnv'")

    args = parser.parse_args()

    #Classifiers I wish to use
    classifiers = StackingClassifier(estimators=[
        	("DecisionTree", DecisionTreeClassifier(class_weight='balanced')),
            ("RandomForest", RandomForestClassifier(class_weight='balanced', n_jobs=-1)),
            ("BalancedRF", BalancedRandomForestClassifier()),
            ("AdaBoost", AdaBoostClassifier()),
            #("ExtraTrees", ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)),
            ("GaussianNB", GaussianNB()),
            #("LDA", LinearDiscriminantAnalysis()),
            ("GradientBoost", GradientBoostingClassifier()),
            #("MLP", MLPClassifier())
        ], cv = 5,
                    stack_method = 'predict_proba',
                    n_jobs=-1,
                    passthrough=False,
                    final_estimator=LogisticRegression(n_jobs = -1),
                    verbose=1)
    
    
    with open("../../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    var = args.var_tag
    if not os.path.exists('../models/'+var):
        os.makedirs('../models/'+var)
    output = "../models/"+var+"/ML_results_"+var+"_.csv"
    #print('Working with '+var+' dataset...', file=open(output, "a"))
    print('Working with '+var+' dataset...')
    X_train, X_test, Y_train, Y_test, background, feature_names = data_parsing(var,config_dict,output)
    with open(output, 'a') as f:
        f.write('Model\ttrain_score\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]\n')
    classifier(classifiers, var, X_train, X_test, Y_train, Y_test,  background, feature_names, output)
    gc.collect()

