#from numpy import mean
import numpy as np
import pandas as pd
import ray
import time
import argparse
import pickle
import yaml
from ray import tune
from ray.tune import Trainable, run
from ray.tune.schedulers.pb2 import PB2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
import os
import gc

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

class RF_PB2(Trainable):  #https://docs.ray.io/en/master/tune/examples/pbt_tune_cifar10_with_keras.html
    def __init__(self, x_train,x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def setup(self, config):
        model = RandomForestClassifier(random_state=42, n_estimators=self.config.get("n_estimators", 100), min_samples_split=self.config.get("min_samples_split",2), min_samples_leaf=self.config.get("min_samples_leaf",1), max_features=self.config.get("max_features","sqrt"), n_jobs = -1)
        #model = RandomForestClassifier(config)
        self.model = model

    def reset_config(self, new_config):
        self.n_estimators = new_config["n_estimators"]
        self.min_samples_split = new_config["min_samples_split"]
        self.min_samples_leaf = new_config["min_samples_leaf"]
        self.max_features = new_config["max_features"]
        self.config = new_config
        return True

    def step(self):
        self.model.fit(self.x_train, self.y_train.ravel())
        y_score = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_score)
        #print(accuracy)
        return {"mean_accuracy": accuracy}

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

    def results(self,config,var, output, feature_names):
        start1 = time.perf_counter()
        clf = RandomForestClassifier(random_state=42, n_estimators=config["n_estimators"], min_samples_split=config["min_samples_split"], min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"], n_jobs = -1)
        score = cross_validate(clf, self.x_train, self.y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
        clf = score['estimator'][np.argmax(score['test_score'])]
        training_score = np.mean(score['train_score'])
        testing_score = np.mean(score['test_score'])
        y_score = clf.predict(self.x_test)
        prc = precision_score(self.y_test,y_score, average="weighted")
        recall  = recall_score(self.y_test,y_score, average="weighted")
        roc_auc = roc_auc_score(self.y_test, y_score)
        accuracy = accuracy_score(self.y_test, y_score)
        matrix = confusion_matrix(self.y_test, y_score)
        finish = (time.perf_counter()-start1)/60
        print(f'RandomForestClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix: {matrix}', file=open(output, "a"))
        # explain all the predictions in the test set
        background = shap.kmeans(self.x_train, 10)
        explainer = shap.KernelExplainer(clf.predict, background)
        with open(f"./tuning/{var}/RandomForestClassifier_{var}.joblib", 'wb') as f:
         dump(clf, f, compress='lz4')
        del clf, self.x_train, self.y_train, background
        background = self.x_test[np.random.choice(self.x_test.shape[0], 1000, replace=False)]
        shap_values = explainer.shap_values(background)
        plt.figure()
        shap.summary_plot(shap_values, background, feature_names, show=False)
        #shap.plots.waterfall(shap_values[0], max_display=15)
        plt.savefig(f"./tuning/{var}/RandomForestClassifier_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
        return None

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
    Y_train = pd.get_dummies(Y_train)
    #Y_train = label_binarize(Y_train.values, classes=['low_impact', 'high_impact']) 

    X_test = pd.read_csv(f'test_{var}/merged_data-test_{var}.csv')
    #var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
    #feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f'test_{var}/merged_data-y-test_{var}.csv')
    print('Data Loaded!')
    Y_test = pd.get_dummies(Y_test)
    #Y_test = label_binarize(Y_test.values, classes=['low_impact', 'high_impact']) 
    print(f'Shape: {Y_test.shape}')
    return X_train, X_test, Y_train, Y_test, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True)
    
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
    with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    pbt = PB2(
        time_attr="training_iteration",
        #metric="mean_accuracy",
        #mode="max",
        perturbation_interval=20,
        #resample_probability=0.25,
        quantile_fraction=0.25,  # copy bottom % with top %
        log_config=True,
        # Specifies the search space for these hyperparams
        hyperparam_bounds={
            "n_estimators" : [50, 200],
            "min_samples_split" : [2, 6],
            "min_samples_leaf" : [1, 4]})
        
    variants = ['non_snv']#,'snv_protein_coding'] #'non_snv','snv',
    for var in variants:
        start = time.perf_counter()
        if not os.path.exists('tuning/'+var):
            os.makedirs('./tuning/'+var)
        output = f"tuning/{var}/RandomForestClassifier_{var}_.csv"
        print('Working with '+var+' dataset...', file=open(output, "w"))
        print('Working with '+var+' dataset...')
        #X_train, X_test, Y_train, Y_test, feature_names = ray.get(data_parsing.remote(var,config_dict,output))
        X_train, X_test, Y_train, Y_test, feature_names = data_parsing(var,config_dict,output)
        #objective = RF_PB2(X_train, X_test, Y_train, Y_test)
        analysis = run(
            RF_PB2(X_train, X_test, Y_train, Y_test),
            name="RandomForestClassifier_PB2",
            verbose=0,
            scheduler=pbt,
            reuse_actors=True,
            #resources_per_trial={
            ##    "cpu": 1,
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
                "training_iteration": 50,
            },
            num_samples=10,
            fail_fast=True,
            queue_trials=True,
            config={ #https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/
                "n_estimators" : tune.randint(50, 200),
                "min_samples_split" : tune.randint(2, 6),
                "min_samples_leaf" : tune.randint(1, 4),
                "max_features" : tune.choice(["sqrt", "log2"])
        })
        finish = (time.perf_counter()- start)/120
        #ttime = (finish- start)/120
        print(f'Total time in hrs: {finish}')
        print("RandomForestClassifier best hyperparameters found were: ", analysis.best_config, file=open(f"tuning/{var}/tuned_parameters_{var}_.csv", "a"))
        RF_PB2(X_train, X_test, Y_train, Y_test).results(analysis.best_config, var, output, feature_names)
        gc.collect()
    

    
    
