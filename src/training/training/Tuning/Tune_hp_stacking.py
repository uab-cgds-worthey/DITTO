import numpy as np
import pandas as pd
import ray
import time
import argparse
import pickle
from joblib import dump
import yaml
#from ray import tune
from ray.tune import Trainable, run
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate  #, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression   #SGDClassifier, 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import os
import gc
import shap
from joblib import dump, load
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')
import functools
print = functools.partial(print, flush=True)

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

def f_unpack_dict(dct):
    '''
    Unpacks all sub-dictionaries in given dictionary recursively. There should be no duplicated keys 
    across all nested subdictionaries, or some instances will be lost without warning
    
    Source: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt
    
    Parameters:
    ----------------
    dct : dictionary to unpack
    
    Returns:
    ----------------
    : unpacked dictionary
    '''
    
    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v
            
    return res

class stacking(Trainable):  #https://docs.ray.io/en/master/hp/examples/pbt_hp_cifar10_with_keras.html
    def setup(self, config, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.config = f_unpack_dict(config)
        self.model = StackingClassifier(estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=self.config.get('rf__n_estimators', 100), criterion=self.config.get('rf__criterion','gini'), max_depth=self.config.get('rf__max_depth', 2), min_samples_split=self.config.get('rf__min_samples_split',2), min_samples_leaf=self.config.get('rf__min_samples_leaf',1), max_features=self.config.get('rf__max_features','sqrt'), oob_score=self.config.get('rf__oob_score',False), class_weight=self.config.get('rf__class_weight','balanced'), n_jobs = -1)),
            #('knn', KNeighborsClassifier(n_neighbors=self.config.get('knn__n_neighbors', 1), weights=self.config.get('knn__weights', 'uniform'), algorithm=self.config.get('knn__algorithm', 'auto'), p=config.get('knn__p', 1), metric=self.config.get('knn__metric', 'minkowski'), n_jobs = -1)),    #leaf_size=self.config.get('leaf_size', 30), 
            ('gbc', GradientBoostingClassifier(random_state=42, loss=self.config.get('gbc__loss', 100), learning_rate = self.config.get('gbc__learning_rate', 0.1), n_estimators=self.config.get('gbc__n_estimators', 100), subsample=self.config.get('gbc__subsample',1), criterion=self.config.get('gbc__criterion','friedman_mse'), min_samples_split=self.config.get('gbc__min_samples_split',2), min_samples_leaf=self.config.get('gbc__min_samples_leaf',1), max_depth=self.config.get('gbc__max_depth', 2), max_features=self.config.get('gbc__max_features','sqrt'))),
            ('dt', DecisionTreeClassifier(random_state=42, criterion=self.config.get('dt__criterion','gini'), splitter=self.config.get('dt__splitter','best'), max_depth=self.config.get('dt__max_depth', 2), min_samples_split=self.config.get('dt__min_samples_split',2), min_samples_leaf=self.config.get('dt__min_samples_leaf',1), max_features=self.config.get('dt__max_features','sqrt'), class_weight=self.config.get('dt__class_weight','balanced'))),
            ('gnb', GaussianNB(var_smoothing=self.config.get('gnb__var_smoothing', 1e-09))),
            ('brf', BalancedRandomForestClassifier(random_state=42, n_estimators=self.config.get('brf__n_estimators', 100), criterion=self.config.get('brf__criterion','gini'), max_depth=self.config.get('brf__max_depth', 2), min_samples_split=self.config.get('brf__min_samples_split',2), min_samples_leaf=self.config.get('brf__min_samples_leaf',1), max_features=self.config.get('brf__max_features','sqrt'), oob_score=self.config.get('brf__oob_score',False), class_weight=self.config.get('brf__class_weight','balanced'), n_jobs = -1)),
            ('lda', LinearDiscriminantAnalysis(solver=self.config.get('lda_solver', 'svd'), shrinkage=self.config.get('lda_shrinkage', None)))
            ],
                    cv = 5,
                    stack_method = 'predict_proba',
                    n_jobs=-1,
                    passthrough=False,
                    final_estimator=LogisticRegression(C=self.config.get('lr__C', 1), penalty=self.config.get('lr__penalty', 'l2'), solver=self.config.get('lr__solver', 'lbfgs'), max_iter=self.config.get('lr__max_iter', 100), l1_ratio=self.config.get('lr__l1_ratio', 0), tol=self.config.get('lr__tol', 1e-4), n_jobs = -1),
                    verbose=0)  #.set_params(**f_unpack_dict(config))
        

    def reset_config(self, new_config):
        self.rf__n_estimators = new_config['rf__n_estimators']
        self.rf__criterion = new_config['rf__criterion']
        self.rf__max_depth = new_config['rf__max_depth']
        self.rf__min_samples_split = new_config['rf__min_samples_split']
        self.rf__min_samples_leaf = new_config['rf__min_samples_leaf']
        self.rf__max_features = new_config['rf__max_features']
        #self.rf__oob_score = new_config['rf__oob_score']
        self.rf__class_weight = new_config['rf__class_weight']
        self.knn__n_neighbors = new_config['knn__n_neighbors']
        self.knn__weights = new_config['knn__weights']
        self.knn__algorithm = new_config['knn__algorithm']
        self.knn__p = new_config['knn__p']
        self.knn__metric = new_config['knn__metric']
        self.gbc__loss = new_config['gbc__loss']
        self.gbc__learning_rate = new_config['gbc__learning_rate']
        self.gbc__n_estimators = new_config['gbc__n_estimators']
        self.gbc__subsample = new_config['gbc__subsample']
        self.gbc__criterion = new_config['gbc__criterion']
        self.gbc__min_samples_split = new_config['gbc__min_samples_split']
        self.gbc__min_samples_leaf = new_config['gbc__min_samples_leaf']
        self.gbc__max_depth = new_config['gbc__max_depth']
        self.gbc__max_features = new_config['gbc__max_features']
        self.dt__criterion = new_config['dt__criterion']
        self.dt__splitter = new_config['dt__splitter']
        self.dt__max_depth = new_config['dt__max_depth']
        self.dt__min_samples_split = new_config['dt__min_samples_split']
        self.dt__min_samples_leaf = new_config['dt__min_samples_leaf']
        self.dt__max_features = new_config['dt__max_features']
        self.dt__class_weight = new_config['dt__class_weight']
        self.gnb__var_smoothing = new_config['gnb__var_smoothing']
        self.brf__n_estimators = new_config['brf__n_estimators']
        self.brf__criterion = new_config['brf__criterion']
        self.brf__max_depth = new_config['brf__max_depth']
        self.brf__min_samples_split = new_config['brf__min_samples_split']
        self.brf__min_samples_leaf = new_config['brf__min_samples_leaf']
        self.brf__max_features = new_config['brf__max_features']
        #self.brf__oob_score = new_config['brf__oob_score']
        self.brf__class_weight = new_config['brf__class_weight']
        self.lda_solver = new_config['lda_solver']
        self.lda_shrinkage = new_config['shrinkage_type_eigen']
        self.lr__C = new_config['lr__C']
        self.lr__solver = new_config['lr__solver']
        self.lr__penalty = new_config['elasticnet']
        self.lr__tol = new_config['lr__tol']
        self.lr__l1_ratio = new_config['lr__l1_ratio']
        self.lr__max_iter = new_config['lr__max_iter']
        self.config = new_config
        return True

    def step(self):
        #score = cross_validate(self.model, self.x_train, self.y_train, cv=3, n_jobs=-1, verbose=0)
        #testing_score = np.max(score['test_score'])
        testing_score = self.model.fit(self.x_train, self.y_train).accuracy_score(self.x_test, self.y_test)
        return {'mean_accuracy': testing_score}

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + '/model'
        pickle.dump(self.model, open(file_path, 'wb'))
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.model
        self.model = pickle.load(open(path,'rb'))

    def cleanup(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print('save model at: ', saved_path)
        pass

def results(config,x_train, x_test, y_train, y_test, var, output, feature_names):
    start1 = time.perf_counter()
    config = f_unpack_dict(config)
    #self.x_train, self.x_test, self.y_train, self.y_test, self.feature_names = self._read_data(config)
    clf = StackingClassifier(estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=config.get('rf__n_estimators', 100), criterion=config.get('rf__criterion','gini'), max_depth=config.get('rf__max_depth', 2), min_samples_split=config.get('rf__min_samples_split',2), min_samples_leaf=config.get('rf__min_samples_leaf',1), max_features=config.get('rf__max_features','sqrt'), oob_score=config.get('rf__oob_score',False), class_weight=config.get('rf__class_weight','balanced'), n_jobs = -1)),
            #('knn', KNeighborsClassifier(n_neighbors=config.get('knn__n_neighbors', 1), weights=config.get('knn__weights', 'uniform'), algorithm=config.get('knn__algorithm', 'auto'), p=config.get('knn__p', 1), metric=config.get('knn__metric', 'minkowski'), n_jobs = -1)),    #leaf_size=config.get('leaf_size', 30), 
            ('gbc', GradientBoostingClassifier(random_state=42, loss=config.get('gbc__loss', 100), learning_rate = config.get('gbc__learning_rate', 0.1), n_estimators=config.get('gbc__n_estimators', 100), subsample=config.get('gbc__subsample',1), criterion=config.get('gbc__criterion','friedman_mse'), min_samples_split=config.get('gbc__min_samples_split',2), min_samples_leaf=config.get('gbc__min_samples_leaf',1), max_depth=config.get('gbc__max_depth', 2), max_features=config.get('gbc__max_features','sqrt'))),
            ('dt', DecisionTreeClassifier(random_state=42, criterion=config.get('dt__criterion','gini'), splitter=config.get('dt__splitter','best'), max_depth=config.get('dt__max_depth', 2), min_samples_split=config.get('dt__min_samples_split',2), min_samples_leaf=config.get('dt__min_samples_leaf',1), max_features=config.get('dt__max_features','sqrt'), class_weight=config.get('dt__class_weight','balanced'))),
            ('gnb', GaussianNB(var_smoothing=config.get('gnb__var_smoothing', 1e-09))),
            ('brf', BalancedRandomForestClassifier(random_state=42, n_estimators=config.get('brf__n_estimators', 100), criterion=config.get('brf__criterion','gini'), max_depth=config.get('brf__max_depth', 2), min_samples_split=config.get('brf__min_samples_split',2), min_samples_leaf=config.get('brf__min_samples_leaf',1), max_features=config.get('brf__max_features','sqrt'), oob_score=config.get('brf__oob_score',False), class_weight=config.get('brf__class_weight','balanced'), n_jobs = -1)),
            ('lda', LinearDiscriminantAnalysis(solver=config.get('lda_solver', 'svd'), shrinkage=config.get('lda_shrinkage', None)))
            ],
                    cv = 5,
                    stack_method = 'predict_proba',
                    n_jobs=-1,
                    passthrough=False,
                    final_estimator= LogisticRegression(C=config.get('lr__C', 1), penalty=config.get('lr__penalty', 'l2'), solver=config.get('lr__solver', 'lbfgs'), max_iter=config.get('lr__max_iter', 100), l1_ratio=config.get('lr__l1_ratio', 0), tol=config.get('lr__tol', 1e-4), n_jobs = -1),
                    verbose=0).fit(x_train, y_train)
    #score = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42), return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
    
    train_score = clf.score(x_train, y_train)
    with open(f'./tuning/{var}/StackingClassifier_{var}.joblib', 'wb') as f:
     dump(clf, f, compress='lz4')
    #training_score = np.mean(score['train_score'])
    #testing_score = np.mean(score['test_score'])
    y_score = clf.predict(x_test)
    prc = precision_score(y_test,y_score, average='weighted')
    recall  = recall_score(y_test,y_score, average='weighted')
    roc_auc = roc_auc_score(y_test, y_score)
    accuracy = accuracy_score(y_test, y_score)
    matrix = confusion_matrix(y_test, y_score)
    finish = (time.perf_counter()-start1)/60
    #print(f'RandomForestClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix:\n{matrix}', file=open(output, 'a'))
    clf_name = str(type(clf)).split("'")[1]  #.split('.')[3]
    print('Model\ttrain_score\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, 'a'))    #\tConfusion_matrix[low_impact, high_impact]
    print(f'{clf_name}\t{train_score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}', file=open(output, 'a'))
    del y_test
    # explain all the predictions in the test set
    background = shap.kmeans(x_train, 10)
    explainer = shap.KernelExplainer(clf.predict, background)
    del clf, x_train, y_train, background
    background = x_test[np.random.choice(x_test.shape[0], 1000, replace=False)]
    del x_test
    shap_values = explainer.shap_values(background)
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names, show=False)
    plt.savefig(f'./tuning/{var}/StackingClassifier_{var}_features.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    del background, explainer, feature_names
    print(f'training and testing done!')
    return None

def wrap_trainable(trainable, x_train, x_test, y_train, y_test):
    x_train_id = ray.put(x_train)
    x_test_id = ray.put(x_test)
    y_train_id = ray.put(y_train)
    y_test_id = ray.put(y_test)

    class _Wrapped(trainable):
        def setup(self, config):
            x_train = ray.get(x_train_id)
            x_test = ray.get(x_test_id)
            y_train = ray.get(y_train_id)
            y_test = ray.get(y_test_id)

            super(_Wrapped, self).setup(config,x_train, x_test, y_train, y_test)

    return _Wrapped

def data_parsing(var,config_dict,output):
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    #Load data
    print(f'\nUsing merged_data-train_{var}..', file=open(output, 'a'))
    X_train = pd.read_csv(f'train_{var}/merged_data-train_{var}.csv')
    #var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict['ML_VAR'], axis=1)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
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
    return X_train, X_test, Y_train, Y_test,feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--smoke-test', action='store_true', help='Finish quickly for testing')
    parser.add_argument(
        '--vtype',
        type=str,
        default='non_snv',
        help='Type of variation/s (without spaces between) to hp the classifier (like: snv,non_snv,snv_protein_coding). (Default: non_snv)')

    args = parser.parse_args()

    variants = args.vtype.split(',')
    
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True)
    
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    with open('../../configs/columns_config.yaml') as fh:
            config_dict = yaml.safe_load(fh)
        
    #variants = ['non_snv','snv','snv_protein_coding']
    for var in variants:
        
        start = time.perf_counter()
        if not os.path.exists('tuning/'+var):
            os.makedirs('./tuning/'+var)
        output = 'tuning/'+var+'/ML_results_'+var+'.csv'
        #print('Working with '+var+' dataset...', file=open(output, 'w'))
        print('Working with '+var+' dataset...')

        x_train, x_test, y_train, y_test, feature_names = data_parsing(var,config_dict,output)

        config={
            #RandomForest - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
                'rf__n_estimators' : hp.randint('rf__n_estimators', 1, 200),
                'rf__criterion' : hp.choice('rf__criterion', ['gini', 'entropy']),
                'rf__max_depth' : hp.randint('rf__max_depth', 2, 200),
                'rf__min_samples_split' : hp.randint('rf__min_samples_split', 2, 10),
                'rf__min_samples_leaf' : hp.randint('rf__min_samples_leaf', 1, 10),
                'rf__max_features' : hp.choice('rf__max_features', ['sqrt', 'log2']),
                #'rf__oob_score' : hp.choice('rf__oob_score', [True, False]),
                'rf__class_weight' : hp.choice('rf__class_weight', ['balanced', 'balanced_subsample']),
            #GradientBoostingClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                'gbc__loss' : hp.choice('gbc__loss', ['deviance', 'exponential']),
                'gbc__learning_rate': hp.loguniform('gbc__learning_rate', 0.01, 1.0),
                'gbc__n_estimators' : hp.randint('gbc__n_estimators', 1, 200),
                'gbc__subsample' : hp.uniform('gbc__subsample', 0.1, 1.0),
                'gbc__criterion' : hp.choice('gbc__criterion', ['friedman_mse', 'mse']),
                'gbc__min_samples_split' : hp.randint('gbc__min_samples_split', 2, 100),
                'gbc__min_samples_leaf' : hp.randint('gbc__min_samples_leaf', 1, 100),
                'gbc__max_depth' : hp.randint('gbc__max_depth', 2, 200),
                'gbc__max_features' : hp.choice('gbc__max_features', ['sqrt', 'log2']),
            #DecisionTree - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
                'dt__criterion' : hp.choice('dt__criterion', ['gini', 'entropy']),
                'dt__splitter' : hp.choice('dt__splitter', ['best', 'random']),
                'dt__max_depth' : hp.randint('dt__max_depth', 2, 200),
                'dt__min_samples_split' : hp.randint('dt__min_samples_split', 2, 100),
                'dt__min_samples_leaf' : hp.randint('dt__min_samples_leaf', 1, 100),
                'dt__max_features' : hp.choice('dt__max_features', ['sqrt', 'log2']),
                'dt__class_weight' : hp.choice('dt__class_weight', [ 'balanced']),
            #GaussianNB - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
                'gnb__var_smoothing' : hp.loguniform('gnb__var_smoothing', 1e-11, 1e-1),
            #BalancedRandomForest - https://imbalanced-learn.org/dev/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
                'brf__n_estimators' : hp.randint('brf__n_estimators', 1, 200),
                'brf__criterion' : hp.choice('brf__criterion', ['gini', 'entropy']),
                'brf__max_depth' : hp.randint('brf__max_depth', 2, 200),
                'brf__min_samples_split' : hp.randint('brf__min_samples_split', 2, 10),
                'brf__min_samples_leaf' : hp.randint('brf__min_samples_leaf', 1, 10),
                'brf__max_features' : hp.choice('brf__max_features', ['sqrt', 'log2']),
                #'brf__oob_score' : hp.choice('brf__oob_score', [True, False]),
                'brf__class_weight' : hp.choice('brf__class_weight', ['balanced', 'balanced_subsample']),
            #LinearDiscriminantAnalysis - https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
                'lda_solver': hp.choice('lda_solver', [
                                     {'lda_solver':'svd'}
                                    ,{'lda_solver':'lsqr','lda_shrinkage':hp.choice('shrinkage_type_lsqr', ['auto', hp.uniform('shrinkage_value_lsqr', 0, 1)])}
                                    ,{'lda_solver':'eigen','lda_shrinkage':hp.choice('shrinkage_type_eigen', ['auto', hp.uniform('shrinkage_value_eigen', 0, 1)])}
                                    ]),
            #LogisticRegression - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic#sklearn.linear_model.LogisticRegression; https://github.com/hyperopt/hyperopt/issues/304
                'lr__C' : hp.uniform('lr__C', 0.0, 100.0),
                'lr__solver':  hp.choice('lr__solver',[
                                        {'lr__solver':'newton-cg', 'lr__penalty': hp.choice('p_newton',['none','l2'])},
                                        {'lr__solver':'lbfgs', 'lr__penalty': hp.choice('p_lbfgs',['none','l2'])},
                                        {'lr__solver': 'liblinear', 'lr__penalty': hp.choice('p_lib',['l1','l2'])}, 
                                        {'lr__solver': 'sag', 'lr__penalty': hp.choice('p_sag',['l2','none'])}, 
                                        {'lr__solver':'saga', 'lr__penalty':'elasticnet', 'lr__l1_ratio': hp.uniform('lr__l1_ratio',0,1)}]),
                'lr__tol': hp.loguniform('lr__tol',1e-13,1e-1),
                
                'lr__max_iter' : hp.randint('lr__max_iter', 2, 100)
        }
        hyperopt_search = HyperOptSearch(config, metric='mean_accuracy', mode='max')
        scheduler = AsyncHyperBandScheduler()

        analysis = run(
            wrap_trainable(stacking, x_train, x_test, y_train, y_test),
            name=f'StackingClassifier_{var}',
            verbose=1,
            scheduler=scheduler,
            search_alg=hyperopt_search,
            reuse_actors=True,
            local_dir='./ray_results',
            max_failures=2,
            #resources_per_trial={
            #    'cpu': 10,
            #    'gpu': 1
            #},
            #global_checkpoint_period=np.inf,   # Do not save checkpoints based on time interval
            checkpoint_freq = 20,        # Save checkpoint every time the checkpoint_score_attr improves
            checkpoint_at_end = True,   
            keep_checkpoints_num = 2,   # Keep only the best checkpoint
            checkpoint_score_attr = 'mean_accuracy', # Metric used to compare checkpoints
            metric='mean_accuracy',
            mode='max',
            stop={
                'training_iteration': 10,
            },
            num_samples=300,
            #fail_fast=True,
            queue_trials=True
        )

        finish = (time.perf_counter()- start)/120
        #ttime = (finish- start)/120
        print(f'Total time in min: {finish}')
        config = analysis.best_config
        print(f'StackingClassifier_{var}:  {config}', file=open(f'tuning/tuned_parameters.csv', 'a'))
        results(config, x_train, x_test, y_train, y_test, var, output, feature_names)
        gc.collect()