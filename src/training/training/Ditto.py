#from numpy import mean
import numpy as np
import pandas as pd
import time
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
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import VotingClassifier
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

@ray.remote
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
    Y_train = label_binarize(Y_train.values, classes=['low_impact', 'high_impact']) 

    X_test = pd.read_csv(f'test_{var}/merged_data-test_{var}.csv')
    #var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
    #feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f'test_{var}/merged_data-y-test_{var}.csv')
    print('Data Loaded!')
    #Y = pd.get_dummies(y)
    Y_test = label_binarize(Y_test.values, classes=['low_impact', 'high_impact']) 
    print(f'Shape: {Y_test.shape}')
    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    # explain all the predictions in the test set
    background = shap.kmeans(X_train, 10)
    return X_train, X_test, Y_train, Y_test, background, feature_names


@ray.remote(num_cpus=9)
def classifier(clf,var, X_train, X_test, Y_train, Y_test,background,feature_names):
   os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
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

   # explain all the predictions in the test set
   #background = shap.kmeans(X_train, 6)
   explainer = shap.KernelExplainer(clf.predict, background)
   clf_name = str(type(clf)).split("'")[1].split(".")[3]
   with open(f"./Ditto/{var}/{clf_name}_{var}.joblib", 'wb') as f:
    dump(clf, f, compress='lz4')
   del clf, X_train, Y_train
   background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
   shap_values = explainer.shap_values(background)
   plt.figure()
   shap.summary_plot(shap_values, background, feature_names, show=False)
   #shap.plots.waterfall(shap_values[0], max_display=15)
   plt.savefig(f"./Ditto/{var}/{clf_name}_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
   finish = (time.perf_counter()-start)/60
   #list1 = [clf_name ,prc, roc_auc, accuracy, score, matrix, finish]
   list1 = [clf_name, np.mean(score['train_score']), np.mean(score['test_score']), prc, recall, roc_auc, accuracy, finish, matrix]
   #pickle.dump(clf, open("./models/"+var+"/"+clf_name+"_"+var+".pkl", 'wb'))
   return list1

if __name__ == "__main__":
    #os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    #Classifiers I wish to use
    classifiers = VotingClassifier(estimators=[
        ('DecisionTreeClassifier',DecisionTreeClassifier(class_weight='balanced')),
        #('SGDClassifier',SGDClassifier(class_weight='balanced', n_jobs=-1)),
        ('RandomForestClassifier',RandomForestClassifier(class_weight='balanced', n_jobs=-1)),
        ('AdaBoostClassifier',AdaBoostClassifier()),
	    ('ExtraTreesClassifier',ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)),
        ('BalancedRandomForestClassifier',BalancedRandomForestClassifier()),
        ('GaussianNB',GaussianNB()),
        ('LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()),
        ('MLPClassifier',MLPClassifier())
        #,('EasyEnsembleClassifier',EasyEnsembleClassifier()) 
        ], voting='hard', n_jobs=-1, verbose=1)
    
    
    with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    variants = ['snv','non_snv','snv_protein_coding'] #'snv',
    for var in variants:
        if not os.path.exists('Ditto/'+var):
            os.makedirs('./Ditto/'+var)
        output = f"Ditto/"+var+"/ML_results_"+var+"_.csv"
        print('Working with '+var+' dataset...', file=open(output, "a"))
        print('Working with '+var+' dataset...')
        X_train, X_test, Y_train, Y_test, background, feature_names = ray.get(data_parsing.remote(var,config_dict,output))
        print('Model\tCross_validate(avg_train_score)\tCross_validate(avg_test_score)\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
        list1 = ray.get(classifier.remote(classifiers,var, X_train, X_test, Y_train, Y_test,background,feature_names))
        print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\t{list1[7]}\n{list1[8]}', file=open(output, "a"))
        print(f'training and testing done!')

