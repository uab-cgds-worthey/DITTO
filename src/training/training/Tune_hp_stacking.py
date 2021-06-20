import numpy as np
import pandas as pd
import ray
import time
import argparse
import pickle
from joblib import dump
import yaml
#from ray import hp
from ray.tune import Trainable, run
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
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
warnings.simplefilter("ignore")
import functools
print = functools.partial(print, flush=True)

hp_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

class stacking(Trainable):  #https://docs.ray.io/en/master/hp/examples/pbt_hp_cifar10_with_keras.html
    def setup(self, config, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = StackingClassifier(estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=self.config.get("rf_n_estimators", 100), criterion=self.config.get("rf_criterion","gini"), max_depth=self.config.get("rf_max_depth", 2), min_samples_split=self.config.get("rf_min_samples_split",2), min_samples_leaf=self.config.get("rf_min_samples_leaf",1), max_features=self.config.get("rf_max_features","sqrt"), oob_score=self.config.get("rf_oob_score",False), class_weight=self.config.get("rf_class_weight","balanced"), n_jobs = -1)),
            ('knn', KNeighborsClassifier(n_neighbors=self.config.get("knn_n_neighbors", 1), weights=self.config.get("knn_weights", 'uniform'), algorithm=self.config.get("knn_algorithm", 'auto'), metric=self.config.get("knn_metric", 'minkowski'), n_jobs = -1)),    #leaf_size=self.config.get("leaf_size", 30), 
            ('gbc', GradientBoostingClassifier(random_state=42, loss=self.config.get("gbc_loss", 100), learning_rate = self.config.get("gbc_learning_rate", 0.1), n_estimators=self.config.get("gbc_n_estimators", 100), subsample=self.config.get("gbc_subsample",1), criterion=self.config.get("gbc_criterion","friedman_mse"), min_samples_split=self.config.get("gbc_min_samples_split",2), min_samples_leaf=self.config.get("gbc_min_samples_leaf",1), max_depth=self.config.get("gbc_max_depth", 2), max_features=self.config.get("gbc_max_features","sqrt"))),
            ('dt', DecisionTreeClassifier(random_state=42, criterion=self.config.get("dt_criterion","gini"), splitter=self.config.get("dt_splitter","best"), max_depth=self.config.get("dt_max_depth", 2), min_samples_split=self.config.get("dt_min_samples_split",2), min_samples_leaf=self.config.get("dt_min_samples_leaf",1), max_features=self.config.get("dt_max_features","sqrt"), class_weight=self.config.get("dt_class_weight","balanced"))),
            #('sgd', SGDClassifier(random_state=42, loss=self.config.get("sgd_loss", "hinge"), penalty=self.config.get("sgd_penalty", "l2"), alpha=self.config.get("sgd_alpha", 0.0001), max_iter=self.config.get("sgd_max_iter", 1000), epsilon=self.config.get("sgd_epsilon", 0.1), learning_rate = self.config.get("sgd_learning_rate", "optimal"), eta0 = self.config.get("sgd_eta0", 0.0), power_t = self.config.get("sgd_power_t", 0.5), class_weight=self.config.get("sgd_class_weight","balanced"), n_jobs = -1)),
            ('gnb', GaussianNB(var_smoothing=self.config.get("var_smoothing", 1e-09))),
            ('brf', BalancedRandomForestClassifier(random_state=42, n_estimators=self.config.get("brf_n_estimators", 100), criterion=self.config.get("brf_criterion","gini"), max_depth=self.config.get("brf_max_depth", 2), min_samples_split=self.config.get("brf_min_samples_split",2), min_samples_leaf=self.config.get("brf_min_samples_leaf",1), max_features=self.config.get("brf_max_features","sqrt"), oob_score=self.config.get("brf_oob_score",False), class_weight=self.config.get("brf_class_weight","balanced"), n_jobs = -1)),
            ('lda', LinearDiscriminantAnalysis(solver=self.config.get("lda_solver", "svd"), shrinkage=self.config.get("lda_shrinkage", None)))
            ],
                    cv = 3,
                    stack_method = "predict_proba",
                    n_jobs=-1,
                    passthrough=False,
                    final_estimator=LogisticRegression(C=self.config.get("lr_C", 1), penalty=self.config.get("lr_penalty", "l2"), solver=self.config.get("lr_solver", "lbfgs"), max_iter=self.config.get("lr_max_iter", 100), n_jobs = -1),
                    verbose=0)
        

    def reset_config(self, new_config):
        self.n_estimators = new_config["rf_n_estimators"]
        self.n_neighbors = new_config["n_neighbors"]
        self.C = new_config["C"]
        self.config = new_config
        return True

    def step(self):
        score = cross_validate(self.model, self.x_train, self.y_train, cv=3, n_jobs=-1, verbose=0)
        testing_score = np.max(score['test_score'])
        #print(accuracy)
        return {"mean_accuracy": testing_score}

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        pickle.dump(self.model, open(file_path, 'wb'))
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.model
        self.model = pickle.load(open(path,'rb'))

    def cleanup(self):
        # If need, save your model when exit.
        # saved_path = self.model.save(self.logdir)
        # print("save model at: ", saved_path)
        pass

def results(config,x_train, x_test, y_train, y_test, var, output, feature_names):
    start1 = time.perf_counter()
    #self.x_train, self.x_test, self.y_train, self.y_test, self.feature_names = self._read_data(config)
    clf = StackingClassifier(estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=config.get("rf_n_estimators", 100), criterion=config.get("rf_criterion","gini"), max_depth=config.get("rf_max_depth", 2), min_samples_split=config.get("rf_min_samples_split",2), min_samples_leaf=config.get("rf_min_samples_leaf",1), max_features=config.get("rf_max_features","sqrt"), oob_score=config.get("rf_oob_score",False), class_weight=config.get("rf_class_weight","balanced"), n_jobs = -1)),
            ('knn', KNeighborsClassifier(n_neighbors=config.get("knn_n_neighbors", 1), weights=config.get("knn_weights", 'uniform'), algorithm=config.get("knn_algorithm", 'auto'), metric=config.get("knn_metric", 'minkowski'), n_jobs = -1)),    #leaf_size=config.get("leaf_size", 30), 
            ('gbc', GradientBoostingClassifier(random_state=42, loss=config.get("gbc_loss", 100), learning_rate = config.get("gbc_learning_rate", 0.1), n_estimators=config.get("gbc_n_estimators", 100), subsample=config.get("gbc_subsample",1), criterion=config.get("gbc_criterion","friedman_mse"), min_samples_split=config.get("gbc_min_samples_split",2), min_samples_leaf=config.get("gbc_min_samples_leaf",1), max_depth=config.get("gbc_max_depth", 2), max_features=config.get("gbc_max_features","sqrt"))),
            ('dt', DecisionTreeClassifier(random_state=42, criterion=config.get("dt_criterion","gini"), splitter=config.get("dt_splitter","best"), max_depth=config.get("dt_max_depth", 2), min_samples_split=config.get("dt_min_samples_split",2), min_samples_leaf=config.get("dt_min_samples_leaf",1), max_features=config.get("dt_max_features","sqrt"), class_weight=config.get("dt_class_weight","balanced"))),
            #('sgd', SGDClassifier(random_state=42, loss=config.get("sgd_loss", "hinge"), penalty=config.get("sgd_penalty", "l2"), alpha=config.get("sgd_alpha", 0.0001), max_iter=config.get("sgd_max_iter", 1000), epsilon=config.get("sgd_epsilon", 0.1), learning_rate = config.get("sgd_learning_rate", "optimal"), eta0 = config.get("sgd_eta0", 0.0), power_t = config.get("sgd_power_t", 0.5), class_weight=config.get("sgd_class_weight","balanced"), n_jobs = -1)),
            ('gnb', GaussianNB(var_smoothing=config.get("var_smoothing", 1e-09))),
            ('brf', BalancedRandomForestClassifier(random_state=42, n_estimators=config.get("brf_n_estimators", 100), criterion=config.get("brf_criterion","gini"), max_depth=config.get("brf_max_depth", 2), min_samples_split=config.get("brf_min_samples_split",2), min_samples_leaf=config.get("brf_min_samples_leaf",1), max_features=config.get("brf_max_features","sqrt"), oob_score=config.get("brf_oob_score",False), class_weight=config.get("brf_class_weight","balanced"), n_jobs = -1)),
            ('lda', LinearDiscriminantAnalysis(solver=config.get("lda_solver", "svd"), shrinkage=config.get("lda_shrinkage", None)))
            ],
                    cv = 3,
                    stack_method = "predict_proba",
                    n_jobs=-1,
                    passthrough=False,
                    final_estimator= LogisticRegression(C=config.get("lr_C", 1), penalty=config.get("lr_penalty", "l2"), solver=config.get("lr_solver", "lbfgs"), max_iter=config.get("lr_max_iter", 100), n_jobs = -1),
                    verbose=0).fit(x_train, y_train)
    #score = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42), return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
    
    train_score = clf.score(x_train, y_train)
    with open(f"./tuning/{var}/StackingClassifier_{var}.joblib", 'wb') as f:
     dump(clf, f, compress='lz4')
    #training_score = np.mean(score['train_score'])
    #testing_score = np.mean(score['test_score'])
    y_score = clf.predict(x_test)
    prc = precision_score(y_test,y_score, average="weighted")
    recall  = recall_score(y_test,y_score, average="weighted")
    roc_auc = roc_auc_score(y_test, y_score)
    accuracy = accuracy_score(y_test, y_score)
    matrix = confusion_matrix(y_test, y_score)
    finish = (time.perf_counter()-start1)/60
    #print(f'RandomForestClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix:\n{matrix}', file=open(output, "a"))
    clf_name = str(type(clf)).split("'")[1]  #.split(".")[3]
    print('Model\ttrain_score\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
    print(f'{clf_name}\t{train_score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}', file=open(output, "a"))
    del y_test
    # explain all the predictions in the test set
    background = shap.kmeans(x_train, 10)
    explainer = shap.KernelExplainer(clf.predict, background)
    del clf, x_train, y_train, background
    background = x_test[np.random.choice(x_test.shape[0], 10, replace=False)]
    del x_test
    shap_values = explainer.shap_values(background)
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names, show=False)
    plt.savefig(f"./tuning/{var}/StackingClassifier_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
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
    print(f'\nUsing merged_data-train_{var}..', file=open(output, "a"))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--vtype",
        type=str,
        default="non_snv",
        help="Type of variation/s (without spaces between) to hp the classifier (like: snv,non_snv,snv_protein_coding). (Default: non_snv)")
    parser.add_argument(
        "--cpus",
        type=int,
        default=10,
        help="Number of CPUs needed. (Default: 10)")
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs needed. (Default: 0)")
    parser.add_argument(
        "--mem",
        type=int,
        default=100*1024*1024*1024,
        help="Memory needed in bytes. (Default: 100*1024*1024*1024 (100GB))")

    args = parser.parse_args()

    variants = args.vtype.split(',')
    
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True, num_cpus=args.cpus, num_gpus=args.gpus, _memory=args.mem)
    
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    with open("../../configs/columns_config.yaml") as fh:
            config_dict = yaml.safe_load(fh)
        
    #variants = ['non_snv','snv','snv_protein_coding']
    for var in variants:
        
        start = time.perf_counter()
        if not os.path.exists('tuning/'+var):
            os.makedirs('./tuning/'+var)
        output = "tuning/"+var+"/ML_results_"+var+".csv"
        #print('Working with '+var+' dataset...', file=open(output, "w"))
        print('Working with '+var+' dataset...')

        x_train, x_test, y_train, y_test, feature_names = data_parsing(var,config_dict,output)

        config={
            #RandomForest - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier
                "rf_n_estimators" : hp.randint('rf_n_estimators', 1, 200),
                "rf_criterion" : hp.choice('rf_criterion', ["gini", "entropy"]),
                "rf_max_depth" : hp.randint('rf_max_depth', 2, 200),
                "rf_min_samples_split" : hp.randint('rf_min_samples_split', 2, 10),
                "rf_min_samples_leaf" : hp.randint('rf_min_samples_leaf', 1, 10),
                "rf_max_features" : hp.choice('rf_max_features', ["sqrt", "log2"]),
                "rf_oob_score" : hp.choice('rf_oob_score', [True, False]),
                "rf_class_weight" : hp.choice('rf_class_weight', ["balanced", "balanced_subsample"]),
            #KNeighborsClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kn#sklearn.neighbors.KNeighborsClassifier
                "knn_n_neighbors" : hp.randint('knn_n_neighbors', 1, 10),
                "knn_weights" : hp.choice('knn_weights', ['uniform', 'distance']),
                "knn_algorithm" : hp.choice('knn_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                "knn_metric" : hp.choice('knn_metric', ['minkowski', 'chebyshev']),
            #GradientBoostingClassifier - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
                "gbc_loss" : hp.choice('gbc_loss', ["deviance", "exponential"]),
                "gbc_learning_rate": hp.loguniform('gbc_learning_rate', 0.01, 1.0),
                "gbc_n_estimators" : hp.randint('gbc_n_estimators', 1, 200),
                "gbc_subsample" : hp.uniform('gbc_subsample', 0.1, 1.0),
                "gbc_criterion" : hp.choice('gbc_criterion', ["friedman_mse", "mse"]),
                "gbc_min_samples_split" : hp.randint('gbc_min_samples_split', 2, 100),
                "gbc_min_samples_leaf" : hp.randint('gbc_min_samples_leaf', 1, 100),
                "gbc_max_depth" : hp.randint('gbc_max_depth', 2, 200),
                "gbc_max_features" : hp.choice('gbc_max_features', ["sqrt", "log2"]),
            #DecisionTree - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
                "dt_criterion" : hp.choice('dt_criterion', ["gini", "entropy"]),
                "dt_splitter" : hp.choice('dt_splitter', ["best", "random"]),
                "dt_max_depth" : hp.randint('dt_max_depth', 2, 200),
                "dt_min_samples_split" : hp.randint('dt_min_samples_split', 2, 100),
                "dt_min_samples_leaf" : hp.randint('dt_min_samples_leaf', 1, 100),
                "dt_max_features" : hp.choice('dt_max_features', ["sqrt", "log2"]),
                "dt_class_weight" : hp.choice('dt_class_weight', [None, "balanced"]),
            #StochasticGradientDescent - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
                #"sgd_loss": hp.choice(['squared_hinge', 'hinge', 'log', 'modified_huber', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                #"sgd_penalty" : hp.choice(['l2', 'l1', 'elasticnet']),
                #"sgd_alpha" : hp.loguniform(1e-9, 1e-1),
                #"sgd_max_iter" : hp.randint(2, 1000),
                #"sgd_epsilon" : hp.uniform(1e-9, 1e-1),
                #"sgd_learning_rate" : hp.choice(['constant', 'optimal', 'invscaling', 'adaptive']),   #'optimal', 
                #"sgd_eta0" : hp.uniform(0.01, 0.9),
                #"sgd_power_t" : hp.uniform(0.1, 0.9),
                #"sgd_class_weight" : hp.choice(["balanced"]),
            #GaussianNB - https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
                "var_smoothing" : hp.loguniform('var_smoothing', 1e-11, 1e-1),
            #BalancedRandomForest - https://imbalanced-learn.org/dev/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html
                "brf_n_estimators" : hp.randint('brf_n_estimators', 1, 200),
                "brf_criterion" : hp.choice('brf_criterion', ["gini", "entropy"]),
                "brf_max_depth" : hp.randint('brf_max_depth', 2, 200),
                "brf_min_samples_split" : hp.randint('brf_min_samples_split', 2, 10),
                "brf_min_samples_leaf" : hp.randint('brf_min_samples_leaf', 1, 10),
                "brf_max_features" : hp.choice('brf_max_features', ["sqrt", "log2"]),
                "brf_oob_score" : hp.choice('brf_oob_score', [True, False]),
                "brf_class_weight" : hp.choice('brf_class_weight', ["balanced", "balanced_subsample"]),
            #LinearDiscriminantAnalysis - https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
                "lda_solver" : hp.choice('lda_solver', ["svd", "lsqr", "eigen"]),
                #"lda_shrinkage" : hp.choice(["auto", None]),
            #LogisticRegression - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic#sklearn.linear_model.LogisticRegression
                "lr_C" : hp.uniform('lr_C', 0.1, 10.0),
                "lr_penalty" : hp.choice('lr_penalty', ["l2"]),
                "lr_solver" : hp.choice('lr_solver', ["newton-cg", "lbfgs", "sag"]),   # "liblinear", "sag", "saga"]),
                "lr_max_iter" : hp.randint('lr_max_iter', 2, 100)
        }
        hyperopt_search = HyperOptSearch(config, metric="mean_accuracy", mode="max")
        scheduler = AsyncHyperBandScheduler()

        analysis = run(
            wrap_trainable(stacking, x_train, x_test, y_train, y_test),
            name=f"StackingClassifier_{var}",
            verbose=1,
            scheduler=scheduler,
            search_alg=hyperopt_search,
            reuse_actors=True,
            local_dir="./ray_results",
            max_failures=2,
            #resources_per_trial={
            #    "cpu": 10,
            #    "gpu": 1
            #},
            #global_checkpoint_period=np.inf,   # Do not save checkpoints based on time interval
            checkpoint_freq = 20,        # Save checkpoint every time the checkpoint_score_attr improves
            checkpoint_at_end = True,   
            keep_checkpoints_num = 2,   # Keep only the best checkpoint
            checkpoint_score_attr = 'mean_accuracy', # Metric used to compare checkpoints
            metric="mean_accuracy",
            mode="max",
            stop={
                "training_iteration": 10,
            },
            num_samples=10,
            #fail_fast=True,
            queue_trials=True
        )

        finish = (time.perf_counter()- start)/120
        #ttime = (finish- start)/120
        print(f'Total time in min: {finish}')
        config = analysis.best_config
        print(f"StackingClassifier_{var}:  {config}", file=open(f"tuning/tuned_parameters.csv", "a"))
        results(config, x_train, x_test, y_train, y_test, var, output, feature_names)
        gc.collect()