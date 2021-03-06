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
import shap
from joblib import dump, load
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
import functools
print = functools.partial(print, flush=True)

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

class DT_PB2(Trainable):  #https://docs.ray.io/en/master/tune/examples/pbt_tune_cifar10_with_keras.html
    def _read_data(self, config):
        os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
        with open("../../configs/columns_config.yaml") as fh:
            config_dict = yaml.safe_load(fh)
        var = config.get("var")
        x_train = pd.read_csv(f'train_{var}/merged_data-train_{var}.csv')
        #var = X_train[config_dict['ML_VAR']]
        x_train = x_train.drop(config_dict['ML_VAR'], axis=1)
        feature_names = x_train.columns.tolist()
        x_train = x_train.values
        y_train = pd.read_csv(f'train_{var}/merged_data-y-train_{var}.csv')
        #Y_train = pd.get_dummies(Y_train)
        y_train = label_binarize(y_train.values, classes=['low_impact', 'high_impact']).ravel()  
        x_test = pd.read_csv(f'test_{var}/merged_data-test_{var}.csv')
        #var = X_test[config_dict['ML_VAR']]
        x_test = x_test.drop(config_dict['ML_VAR'], axis=1)
        #feature_names = X_test.columns.tolist()
        x_test = x_test.values
        y_test = pd.read_csv(f'test_{var}/merged_data-y-test_{var}.csv')
        print('Data Loaded!')
        #Y_test = pd.get_dummies(Y_test)
        y_test = label_binarize(y_test.values, classes=['low_impact', 'high_impact']).ravel()  
        #print(f'Shape: {Y_test.shape}')
        return x_train, x_test, y_train, y_test, feature_names

    def setup(self, config):
        self.x_train, self.x_test, self.y_train, self.y_test, self.feature_names = self._read_data(config)
        model = DecisionTreeClassifier(random_state=42, min_samples_split=self.config.get("min_samples_split",2), min_samples_leaf=self.config.get("min_samples_leaf",1), criterion=self.config.get("criterion","gini"), max_features=self.config.get("max_features","sqrt"), class_weight=self.config.get("class_weight","balanced"))
        #model = RandomForestClassifier(config)
        self.model = model

    def reset_config(self, new_config):
        self.min_samples_split = new_config["min_samples_split"]
        self.min_samples_leaf = new_config["min_samples_leaf"]
        self.criterion = new_config["criterion"]
        self.max_features = new_config["max_features"]
        self.class_weight = new_config["class_weight"]
        self.config = new_config
        return True

    def step(self):
        score = cross_validate(self.model, self.x_train, self.y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0)
        testing_score = np.mean(score['test_score'])
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

    def results(self,config,var, output):
        start1 = time.perf_counter()
        self.x_train, self.x_test, self.y_train, self.y_test, self.feature_names = self._read_data(config)
        clf = DecisionTreeClassifier(random_state=42, min_samples_split=config["min_samples_split"], min_samples_leaf=config["min_samples_leaf"], criterion=config["criterion"], max_features=config["max_features"], class_weight=config["class_weight"])
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
        print(f'DecisionTreeClassifier: \nCross_validate(avg_train_score): {training_score}\nCross_validate(avg_test_score): {testing_score}\nPrecision: {prc}\nRecall: {recall}\nROC_AUC: {roc_auc}\nAccuracy: {accuracy}\nTime(in min): {finish}\nConfusion Matrix:\n{matrix}', file=open(output, "a"))
        # explain all the predictions in the test set
        background = shap.kmeans(self.x_train, 10)
        explainer = shap.KernelExplainer(clf.predict, background)
        with open(f"./tuning/{var}/DecisionTreeClassifier_{var}.joblib", 'wb') as f:
         dump(clf, f, compress='lz4')
        del clf, self.x_train, self.y_train, background
        background = self.x_test[np.random.choice(self.x_test.shape[0], 1000, replace=False)]
        shap_values = explainer.shap_values(background)
        plt.figure()
        shap.summary_plot(shap_values, background, self.feature_names, show=False)
        plt.savefig(f"./tuning/{var}/DecisionTreeClassifier_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
        return None

if __name__ == "__main__":
    variants = ['non_snv','snv','snv_protein_coding']
    for var in variants:    
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--smoke-test", action="store_true", help="Finish quickly for testing")
        args, _ = parser.parse_known_args()
        if args.smoke_test:
            ray.init(num_cpus=2)  # force pausing to happen for test
        else:
            ray.init(ignore_reinit_error=True)

        os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')


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
                "min_samples_split" : [2, 6],
                "min_samples_leaf" : [1, 4]})
        
    
        start = time.perf_counter()
        if not os.path.exists('tuning/'+var):
            os.makedirs('./tuning/'+var)
        output = f"tuning/{var}/DecisionTreeClassifier_{var}_.csv"
        print('Working with '+var+' dataset...', file=open(output, "w"))
        print('Working with '+var+' dataset...')
        analysis = run(
            DT_PB2,
            name=f"DecisionTreeClassifier_PB2_{var}",
            verbose=0,
            scheduler=pbt,
            reuse_actors=True,
            local_dir="./tune_results",
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
                "training_iteration": 100,
            },
            num_samples=5,
            fail_fast=True,
            queue_trials=True,
            config={ #https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/
                "var": var,
                "min_samples_split" : tune.randint(2, 6),
                "min_samples_leaf" : tune.randint(1, 4),
                "criterion" : tune.choice(["gini", "entropy"]),
                "max_features" : tune.choice(["sqrt", "log2"]),
                "class_weight" : tune.choice(["balanced"])
        })
        finish = (time.perf_counter()- start)/120
        #ttime = (finish- start)/120
        print(f'Total time in min: {finish}')
        print(f"DecisionTreeClassifier best hyperparameters found for {var} were:  {analysis.best_config}", file=open(f"tuning/tuned_parameters.csv", "a"))
        DT_PB2(analysis.best_config).results(analysis.best_config, var, output)
        gc.collect()
    