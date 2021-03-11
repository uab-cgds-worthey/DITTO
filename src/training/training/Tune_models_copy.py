#from numpy import mean
import numpy as np
import pandas as pd
import time
#import ray
## Start Ray.
#ray.init(ignore_reinit_error=True)
import warnings
warnings.simplefilter("ignore")
import joblib
from ray.util.joblib import register_ray
from joblib import dump, load
import shap
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.ensemble import  ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import yaml
import functools
print = functools.partial(print, flush=True)
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

##Patch to fix the version of sklearn. Refer this - https://github.com/scikit-optimize/scikit-optimize/issues/978
#def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
#                         n_iter=50, scoring=None, fit_params=None, n_jobs=1,
#                         n_points=1, iid=True, refit=True, cv=None, verbose=0,
#                         pre_dispatch='2*n_jobs', random_state=None,
#                         error_score='raise', return_train_score=False):
#
#        self.search_spaces = search_spaces
#        self.n_iter = n_iter
#        self.n_points = n_points
#        self.random_state = random_state
#        self.optimizer_kwargs = optimizer_kwargs
#        self._check_search_space(self.search_spaces)
#        self.fit_params = fit_params
#
#        super(BayesSearchCV, self).__init__(
#             estimator=estimator, scoring=scoring,
#             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
#             pre_dispatch=pre_dispatch, error_score=error_score,
#             return_train_score=return_train_score)
#        
#BayesSearchCV.__init__ = bayes_search_CV_init

#@ray.remote
def data_parsing(var,config_dict,output):
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    #Load data
    print(f'\nUsing merged_data-train_{var}..', file=open(output, "a"))
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
    #background = shap.kmeans(X_train, 10)
    return X_train, X_test, Y_train, Y_test, feature_names


#@ray.remote
def classifier(clf,model,var, X_train, X_test, Y_train, Y_test,feature_names):
   os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
   start = time.perf_counter()
   #score = cross_validate(clf, X_train, Y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0, scoring=('roc_auc','neg_log_loss'))
   #clf = score['estimator'][np.argmin(score['test_neg_log_loss'])]
   #y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
   #class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
   #clf.fit(X_train, Y_train) #, class_weight=class_weights)
   #clf_name = str(type(clf)).split("'")[1]  #.split(".")[3]
   with open(f"./tuning/{var}/{model}_{var}.joblib", 'wb') as f:
    dump(clf, f, compress='lz4')
   #del clf
   #with open(f"./models/{var}/{clf_name}_{var}.joblib", 'rb') as f:
   # clf = load(f)
   y_score = clf.predict(X_test)
   prc = precision_score(Y_test,y_score, average="weighted")
   recall  = recall_score(Y_test,y_score, average="weighted")
   roc_auc = roc_auc_score(Y_test, y_score)
   #roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
   accuracy = accuracy_score(Y_test, y_score)
   score = clf.score(X_train, Y_train)
   #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
   matrix = confusion_matrix(Y_test, y_score)

   # explain all the predictions in the test set
   #background = shap.kmeans(X_train, 6)
   #explainer = shap.KernelExplainer(clf.predict, background)
   del clf, X_train
   #background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
   #shap_values = explainer.shap_values(background)
   #plt.figure()
   #shap.summary_plot(shap_values, background, feature_names, show=False)
   ##shap.plots.waterfall(shap_values[0], max_display=15)
   #plt.savefig(f"./models/{var}/{clf_name}_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
   finish = (time.perf_counter()-start)/60
   list1 = [model ,prc, recall, roc_auc, accuracy, score, finish, matrix]
   #list1 = [clf_name, np.mean(score['train_roc_auc']), np.mean(score['test_roc_auc']),np.mean(score['train_neg_log_loss']), np.mean(score['test_neg_log_loss']), prc, recall, roc_auc, accuracy, finish, matrix]
   #pickle.dump(clf, open("./models/"+var+"/"+clf_name+"_"+var+".pkl", 'wb'))
   return list1


if __name__ == "__main__":
    #Classifiers I wish to use
    classifiers = {
        	ExtraTreesClassifier():{ #bootstrap = True,
                #  warm_start=True, 
                #  oob_score=True): {
                "n_estimators" : Integer(50, 200),
                "min_samples_split" : Integer(2, 100),
                "min_samples_leaf" : Integer(1, 100),
                "criterion" : Categorical(["gini", "entropy"]),
                "max_features" : Categorical(["sqrt", "log2"]),
                "class_weight" : Categorical([None, "balanced", "balanced_subsample"])
            },
            DecisionTreeClassifier(): {
                "min_samples_split" : Integer(2, 100),
                "min_samples_leaf" : Integer(1, 100),
                "criterion" : Categorical(["gini", "entropy"]),
                "max_features" : Categorical(["sqrt", "log2"]),
                "class_weight" : Categorical([None, "balanced"])
                },
            SGDClassifier(): {
                'loss': Categorical(['squared_hinge', 'hinge']),
                'alpha': Real(1e-4, 1e-1),
                'epsilon': Real(1e-2, 1e-1)
            },
            RandomForestClassifier(n_jobs=-1): {
                "n_estimators" : Integer(10, 200),
                "min_samples_split" : Integer(2, 100),
                "min_samples_leaf" : Integer(1, 100),
                "criterion" : Categorical(["gini", "entropy"]),
                "max_features" : Categorical(["sqrt", "log2"]),
                "class_weight" : Categorical(["balanced", "balanced_subsample"]),
                "oob_score" : Categorical([True, False]),
                "max_depth" : Integer(1, 200)
                            },
            #AdaBoostClassifier(),
            
            #BalancedRandomForestClassifier(),
            #GaussianNB(),
            #LinearDiscriminantAnalysis(),
            GradientBoostingClassifier(): {
                "n_estimators" : Integer(10, 200),
                "max_depth" : Integer(1, 200),
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                "min_samples_split" : Integer(2, 100),
                "min_samples_leaf" : Integer(1, 100),
                "max_features" : Integer(1, 10),
                "subsample" : Real(0.0, 1.0),
                "max_features" : Categorical(["sqrt", "log2"])
            },
            #MLPClassifier()
    }
    
    
    with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    #variants = ['snv','non_snv','snv_protein_coding'] #'snv',
    variants = ['non_snv']
    for var in variants:
        if not os.path.exists('tuning/'+var):
            os.makedirs('./tuning/'+var)
        output = "tuning/"+var+"/ML_results_"+var+"_.csv"
        print('Working with '+var+' dataset...', file=open(output, "w"))
        print('Working with '+var+' dataset...')
        #X_train, X_test, Y_train, Y_test, feature_names = ray.get(data_parsing.remote(var,config_dict,output))
        X_train, X_test, Y_train, Y_test, feature_names = data_parsing(var,config_dict,output)
        #print('Model\tCross_validate(avg_train_roc_auc)\tCross_validate(avg_test_roc_auc)\tCross_validate(avg_train_neg_log_loss)\tCross_validate(avg_test_neg_log_loss)\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
        print('Model\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tScore\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
        for model, config in zip(classifiers.keys(), classifiers.values()):
            clf = BayesSearchCV(model, config, n_jobs=-1,
                        cv=10,
                        random_state=42, n_iter=200, refit=True)
            register_ray()
            with joblib.parallel_backend('ray'):
                clf.fit(X_train, Y_train)
            print(f'{model}_{var}:{clf.best_params_}', file=open("tuning/tuned_parameters.csv", "a"))
            list1 = classifier(clf,model, var, X_train, X_test, Y_train, Y_test,feature_names)
            #print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\t{list1[7]}\t{list1[8]}\t{list1[9]}\n{list1[10]}', file=open(output, "a"))
            print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\n{list1[7]}', file=open(output, "a"))
            print(f'{model} training and testing done!')
            del clf

