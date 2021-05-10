import numpy as np
import pandas as pd
import ray
import time
import argparse
import pickle
from joblib import dump, load
import yaml
from ray import tune
from ray.tune import Trainable, run
from ray.tune.suggest.skopt import SkOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
from sklearn.ensemble import RandomForestClassifier
import os
import gc
import shap
from joblib import dump, load
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import functools
print = functools.partial(print, flush=True)

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

class RF_PB2(Trainable):  #https://docs.ray.io/en/master/tune/examples/pbt_tune_cifar10_with_keras.html
    def setup(self, config, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = RandomForestClassifier(random_state=42, n_estimators=self.config.get("n_estimators", 100), max_depth=self.config.get("max_depth", 2), min_samples_split=self.config.get("min_samples_split",2), min_samples_leaf=self.config.get("min_samples_leaf",1), criterion=self.config.get("criterion","gini"), max_features=self.config.get("max_features","sqrt"), class_weight=self.config.get("class_weight","balanced"), n_jobs = -1)
        

    def reset_config(self, new_config):
        self.n_estimators = new_config["n_estimators"]
        self.min_samples_split = new_config["min_samples_split"]
        self.min_samples_leaf = new_config["min_samples_leaf"]
        self.criterion = new_config["criterion"]
        self.max_features = new_config["max_features"]
        self.class_weight = new_config["class_weight"]
        #self.oob_score = new_config["oob_score"]
        self.max_depth = new_config["max_depth"]
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
    clf = RandomForestClassifier(random_state=42, n_estimators=config.get("n_estimators", 100), max_depth=config.get("max_depth", 2), min_samples_split=config.get("min_samples_split",2), min_samples_leaf=config.get("min_samples_leaf",1), criterion=config.get("criterion","gini"), max_features=config.get("max_features","sqrt"), class_weight=config.get("class_weight","balanced"), n_jobs = -1)
    score = cross_validate(clf, x_train, y_train, cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42), return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
    clf = score['estimator'][np.argmax(score['test_score'])]
    with open(f"./tuning/{var}/RandomForestClassifier_{var}.joblib", 'wb') as f:
     dump(clf, f, compress='lz4')
    training_score = np.mean(score['train_score'])
    testing_score = np.mean(score['test_score'])
    y_score = clf.predict(x_test)
    prc = precision_score(y_test,y_score, average="weighted")
    recall  = recall_score(y_test,y_score, average="weighted")
    roc_auc = roc_auc_score(y_test, y_score)
    accuracy = accuracy_score(y_test, y_score)
    matrix = confusion_matrix(y_test, y_score)
    finish = (time.perf_counter()-start1)/60
    #print(f'RandomForestClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix:\n{matrix}', file=open(output, "a"))
    clf_name = str(type(clf)).split("'")[1]  #.split(".")[3]
    print('Model\tCross_validate(avg_train_score)\tCross_validate(avg_test_score)\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
    print(f'{clf_name}\t{training_score}\t{testing_score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}', file=open(output, "a"))
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
    plt.savefig(f"./tuning/{var}/RandomForestClassifier_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    del background, explainer, feature_names
    print(f'{clf} training and testing done!')
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
        help="Type of variation/s (without spaces between) to tune the classifier (like: snv,non_snv,snv_protein_coding). (Default: non_snv)")

    args = parser.parse_args()

    variants = args.vtype.split(',')
    
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True)
    
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

        skopt_search = SkOptSearch(
            metric="mean_accuracy",
            mode="max")
        skopt_search = ConcurrencyLimiter(skopt_search, max_concurrent=10)
        scheduler = AsyncHyperBandScheduler()

        analysis = run(
            wrap_trainable(RF_PB2, x_train, x_test, y_train, y_test),
            name=f"RandomForestClassifier_PB2_{var}",
            verbose=1,
            scheduler=scheduler,
            search_alg=skopt_search,
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
            keep_checkpoints_num = 1,   # Keep only the best checkpoint
            checkpoint_score_attr = 'mean_accuracy', # Metric used to compare checkpoints
            metric="mean_accuracy",
            mode="max",
            stop={
                "training_iteration": 100,
            },
            num_samples=5,
            #fail_fast=True,
            queue_trials=True,
            config={ #https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/
                "n_estimators" : tune.randint(1, 200),
                "min_samples_split" : tune.randint(2, 10),
                "min_samples_leaf" : tune.randint(1, 10),
                "criterion" : tune.choice(["gini", "entropy"]),
                "max_features" : tune.choice(["sqrt", "log2"]),
                "class_weight" : tune.choice(["balanced", "balanced_subsample"]),
                #"oob_score" : tune.choice([True, False]),
                "max_depth" : tune.randint(2, 200)

        })
        finish = (time.perf_counter()- start)/120
        #ttime = (finish- start)/120
        print(f'Total time in min: {finish}')
        config = analysis.best_config
        print(f"RandomForestClassifier_{var}:  {config}", file=open(f"tuning/tuned_parameters.csv", "a"))
        results(config, x_train, x_test, y_train, y_test, var, output, feature_names)
        gc.collect()