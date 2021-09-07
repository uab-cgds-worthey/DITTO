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
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
        self.model = LinearDiscriminantAnalysis(solver=self.config.get('lda_solver', 'svd'), shrinkage=self.config.get('lda_shrinkage', None))
        

    def reset_config(self, new_config):
        self.lda_solver = new_config['lda_solver']
        self.lda_shrinkage = new_config['lda_shrinkage']
        self.config = new_config
        return True

    def step(self):
        score = cross_validate(self.model, self.x_train, self.y_train, cv=5, n_jobs=-1, verbose=0)
        testing_score = np.max(score['test_score'])
        #testing_score = self.model.fit(self.x_train, self.y_train).accuracy_score(self.x_test, self.y_test)
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
    clf_name = "LinearDiscriminantAnalysis"
    clf = LinearDiscriminantAnalysis(solver=config.get('lda_solver', 'svd'), shrinkage=config.get('lda_shrinkage', None))
    #score = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42), return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
    clf.fit(x_train,y_train)
    train_score = clf.score(x_train, y_train)
    with open(f'../tuning/{var}/{clf_name}_tuned_{var}.joblib', 'wb') as f:
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
    #with open(output, 'a') as f:
    #    f.write(f"AdaBoostClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix:\n{matrix}")
    
    print(f'Model\ttrain_score\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, 'a'))    #\tConfusion_matrix[low_impact, high_impact]
    print(f'{clf_name}\t{train_score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}', file=open(output, 'a'))
    del y_test
    # explain all the predictions in the test set
    background = shap.kmeans(x_train, 10)
    explainer = shap.KernelExplainer(clf.predict, background)
    del clf, x_train, y_train, background
    background = x_test[np.random.choice(x_test.shape[0], 10000, replace=False)]
    del x_test
    shap_values = explainer.shap_values(background)
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names, show=False)
    plt.savefig(f'../tuning/{var}/{clf_name}_tuned_{var}_features.pdf', format='pdf', dpi=1000, bbox_inches='tight')
    del background, explainer, feature_names
    print(f'tuning and testing done!')
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
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test')
    #Load data
    #print(f'\nUsing merged_data-train_{var}..', file=open(output, 'a'))
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
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        default="nssnv",
        help="Var tag used while filtering or untuned models. (Default: nssnv)")

    args = parser.parse_args()

    var = args.var_tag
    
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True)
    
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test')
    with open('../../../configs/columns_config.yaml') as fh:
            config_dict = yaml.safe_load(fh)
        
    start = time.perf_counter()
    if not os.path.exists('../tuning/'+var):
        os.makedirs('../tuning/'+var)
    output = '../tuning/'+var+'/ML_results_'+var+'.csv'
    #print('Working with '+var+' dataset...', file=open(output, 'w'))
    print('Working with '+var+' dataset...')
    x_train, x_test, y_train, y_test, feature_names = data_parsing(var,config_dict,output)
    config={
        'lda_solver': hp.choice('lda_solver', [
                                     {'lda_solver':'svd'},
                                     {'lda_solver':'lsqr','lda_shrinkage':hp.choice('shrinkage_type_lsqr', ['auto', hp.uniform('shrinkage_value_lsqr', 0, 1)])}
                                    ,{'lda_solver':'eigen','lda_shrinkage':hp.choice('shrinkage_type_eigen', ['auto', hp.uniform('shrinkage_value_eigen', 0, 1)])}
                                    ]),
    }
    hyperopt_search = HyperOptSearch(config, metric='mean_accuracy', mode='max')
    #scheduler = AsyncHyperBandScheduler()
    analysis = run(
        wrap_trainable(stacking, x_train, x_test, y_train, y_test),
        name=f'LinearDiscriminantAnalysis_{var}',
        verbose=1,
        #scheduler=scheduler,
        search_alg=hyperopt_search,
        reuse_actors=True,
        local_dir='../ray_results',
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
            'training_iteration': 1,
        },
        num_samples=500,
        #fail_fast=True,
        queue_trials=True
    )
    finish = (time.perf_counter()- start)/120
    #ttime = (finish- start)/120
    print(f'Total time in min: {finish}')
    config = analysis.best_config
    print(f'LinearDiscriminantAnalysis_{var}:  {config}', file=open(f'../tuning/tuned_parameters.csv', 'a'))
    results(config, x_train, x_test, y_train, y_test, var, output, feature_names)
    gc.collect()