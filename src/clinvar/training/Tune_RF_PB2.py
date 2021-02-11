#from numpy import mean
import numpy as np
import pandas as pd
import ray
import time
import argparse
import pickle
from ray import tune
from ray.tune import Trainable
from ray.tune.schedulers.pb2 import PB2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import os

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

class Ditto(Trainable):  #https://docs.ray.io/en/master/tune/examples/pbt_tune_cifar10_with_keras.html
    def _read_data(self):
        os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')
        X = pd.read_csv('clinvar-md.csv')
        var = X[['AAChange.refGene','ID']]
        X = X.drop(['AAChange.refGene', 'ID'], axis=1)
        X = X.values
        # X[1]
        # var
        y = pd.read_csv('clinvar-y-md.csv')
        #Y = pd.get_dummies(y)
        Y = label_binarize(y.values, classes=['Benign', 'Pathogenic']) #'Benign', 'Likely_benign', 'Uncertain_significance', 'Likely_pathogenic', 'Pathogenic'
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.30, random_state=42)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        return (X_train, Y_train), (X_test, Y_test)

    def setup(self, config):
        self.train_data, self.test_data = self._read_data()
        x_train = self.train_data[0]
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
        x_train, y_train = self.train_data
        #x_train, y_train = x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
        x_test, y_test = self.test_data
        #x_test, y_test = x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]
        self.model.fit(x_train, y_train.ravel())
        y_score = self.model.predict_proba(x_test)
        accuracy = average_precision_score(y_test, np.argmax(y_score, axis=1), average=None)
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

    def results(self,config):
        X_train, Y_train = self.train_data
        #x_train, y_train = x_train[:NUM_SAMPLES], y_train[:NUM_SAMPLES]
        X_test, Y_test = self.test_data
        #x_test, y_test = x_test[:NUM_SAMPLES], y_test[:NUM_SAMPLES]
        start = time.perf_counter()
        clf = RandomForestClassifier(random_state=42, n_estimators=config["n_estimators"], min_samples_split=config["min_samples_split"], min_samples_leaf=config["min_samples_leaf"], max_features=config["max_features"], n_jobs = -1)
        clf.fit(X_train, Y_train)
        y_score = clf.predict_proba(X_test)
        prc = average_precision_score(Y_test, np.argmax(y_score, axis=1), average=None)
        score = clf.score(X_train, Y_train)
        #matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
        matrix = confusion_matrix(Y_test, np.argmax(y_score, axis=1))
        finish = (time.perf_counter()-start)/60
        list1 = [clf ,prc, score, matrix, finish]
        print('Model\tprecision_score\tTrain_score\tTime(min)\tConfusion_matrix[Benign, Pathogenic]')
        print(f'{clf}\n{prc}\t{score}\t{finish}\n{matrix}')
        return clf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init()
   
     

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
        

    analysis = tune.run(
        Ditto,
        name="Ditto_RF",
        verbose=0,
        scheduler=pbt,
        reuse_actors=True,
        resources_per_trial={
        #    "cpu": 1,
            "gpu": 1
        },
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
        num_samples=30,
        fail_fast=True,
        config={ #https://www.geeksforgeeks.org/hyperparameters-of-random-forest-classifier/
            "n_estimators" : tune.randint(50, 200),
            "min_samples_split" : tune.randint(2, 6),
            "min_samples_leaf" : tune.randint(1, 4),
            "max_features" : tune.choice(["sqrt", "log2"])
        })

    print("Model: RandomForest(sklearn)")
    #config = analysis.best_config
    print("Best hyperparameters found were: ", analysis.best_config)
    
    clf = Ditto().results(analysis.best_config)
    pickle.dump(clf, open("./models/Randomforest.pkl", 'wb'))
    
