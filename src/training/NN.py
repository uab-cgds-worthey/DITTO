#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 01:11:09 2020

@author: tarunmamidi
"""
import time
import numpy as np

np.random.seed(5)
from pathlib import Path
import optuna
from optuna.integration import TFKerasPruningCallback
from optuna.integration.tensorboard import TensorBoardCallback
from optuna.samplers import TPESampler
import tensorflow as tf
import argparse
import os
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
try:
    tf.get_logger().setLevel("INFO")
except Exception as exc:
    print(exc)
import warnings

warnings.simplefilter("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
)
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import shap

# from joblib import dump, load


# EPOCHS = 150
class Objective(object):
    def __init__(self, train_x, test_x, train_y, test_y, class_weights):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.class_weights = class_weights

    def __call__(self, config):
        # Clear clutter from previous TensorFlow graphs.
        tf.keras.backend.clear_session()

        # Metrics to be monitored by Optuna.
        if tf.__version__ >= "2":
            monitor = "val_accuracy"
        else:
            monitor = "val_acc"
        n_layers = config.suggest_int("n_layers", 1, 30)
        model = Sequential()
        model.add(
            Dense(
                self.train_x.shape[1],
                input_shape=(self.train_x.shape[1],),
                activation=config.suggest_categorical(
                    "activation",
                    [
                        "tanh",
                        "softmax",
                        "elu",
                        "softplus",
                        "softsign",
                        "relu",
                        "sigmoid",
                        "hard_sigmoid",
                        "linear",
                    ],
                ),
            )
        )
        for i in range(n_layers):
            num_hidden = config.suggest_int("n_units_l{}".format(i), 1, 200)
            model.add(
                Dense(
                    num_hidden,
                    name="dense_l{}".format(i),
                    kernel_initializer=config.suggest_categorical(
                        "kernel_initializer_l{}".format(i),
                        [
                            "uniform",
                            "lecun_uniform",
                            "normal",
                            "zero",
                            "glorot_normal",
                            "glorot_uniform",
                            "he_normal",
                            "he_uniform",
                        ],
                    ),
                    activation=config.suggest_categorical(
                        "activation_l{}".format(i),
                        [
                            "tanh",
                            "softmax",
                            "elu",
                            "softplus",
                            "softsign",
                            "relu",
                            "sigmoid",
                            "hard_sigmoid",
                            "linear",
                        ],
                    ),
                )
            )
            model.add(
                Dropout(
                    config.suggest_float("dropout_l{}".format(i), 0.0, 0.9),
                    name="dropout_l{}".format(i),
                )
            )
        model.add(
            Dense(
                units=len(np.unique(self.train_y)),
                name="dense_last",
                kernel_initializer=config.suggest_categorical(
                    "kernel_initializer",
                    [
                        "uniform",
                        "lecun_uniform",
                        "normal",
                        "zero",
                        "glorot_normal",
                        "glorot_uniform",
                        "he_normal",
                        "he_uniform",
                    ],
                ),
                activation="sigmoid",
            )
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=config.suggest_categorical(
                "optimizer",
                ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"],
            ),
            metrics=["accuracy"],
        )
        # model.summary()
        # Create callbacks for early stopping and pruning.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10),
            TFKerasPruningCallback(config, monitor),
        ]

        # Train the model
        history = model.fit(
            self.train_x,
            self.train_y,
            validation_split=0.2,
            verbose=0,
            shuffle=True,
            callbacks=callbacks,
            batch_size=config.suggest_int("batch_size", 10, 1000),
            epochs=500,
            class_weight = self.class_weights
        )
        return history.history["val_accuracy"][-1]
        # Evaluate the model accuracy on the validation set.
        # score = model.evaluate(self.val_x, self.val_y, verbose=0)
        # return score[1]

    def tuned_run(self, config):
        # Clear clutter from previous TensorFlow graphs.
        print("running tuned params\n")
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(
            Dense(
                self.train_x.shape[1],
                input_shape=(self.train_x.shape[1],),
                activation=config["activation"],
            )
        )
        for i in range(config["n_layers"]):
            model.add(
                Dense(
                    config["n_units_l{}".format(i)],
                    name="dense_l{}".format(i),
                    kernel_initializer=config["kernel_initializer_l{}".format(i)],
                    activation=config["activation_l{}".format(i)],
                )
            )
            model.add(Dropout(config["dropout_l{}".format(i)]))
        model.add(
            Dense(
                units=len(np.unique(self.train_y)),
                name="dense_last",
                kernel_initializer=config["kernel_initializer"],
                activation="sigmoid",
            )
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=config["optimizer"],
            metrics=["accuracy"],
        )
        # model.summary()
        # Train the model
        model.fit(
            self.train_x,
            self.train_y,
            verbose=2,
            batch_size=config["batch_size"],
            epochs=500,
            class_weights = self.class_weights
        )
        # Evaluate the model accuracy on the validation set.
        # score = model.evaluate(test_x, test_y, verbose=0)
        return model

    def show_result(self, study, out_dir, output, feature_names):
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print(
            f"NeuralNetwork:  {trial.params}",
            file=open(out_dir + "/tuned_parameters.csv", "a"),
        )

        model = self.tuned_run(trial.params)
        print("ran tuned model\n")
        results = model.evaluate(self.test_x, self.test_y)
        y_score = model.predict(self.test_x)
        prc = precision_score(self.test_y, y_score.round(), average="weighted")
        recall = recall_score(self.test_y, y_score.round(), average="weighted")
        roc_auc = roc_auc_score(self.test_y, y_score.round())
        accuracy = accuracy_score(self.test_y, y_score.round())
        # prc_micro = average_precision_score(self.test_y, y_score, average='micro')
        matrix = confusion_matrix(np.argmax(self.test_y.values, axis=1), np.argmax(y_score, axis=1))
        print(
            f"Model\tTest_loss\tTest_accuracy\tPrecision\tRecall\troc_auc\tAccuracy\tConfusion_matrix[low_impact, high_impact]",
            file=open(output, "a"),
        )  # \tConfusion_matrix[low_impact, high_impact]
        print(
            f"Neural_Network\t{results[0]}\t{results[1]}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\n{matrix}",
            file=open(output, "a"),
        )  # results:\nstorage ="sqlite:///../tuning/{var}/Neural_network_{var}.db"
        # Calling `save('my_model')` creates a SavedModel folder `my_model`.
        model.save(out_dir + "/Neural_network")
        model.save_weights(out_dir + "/weights.h5")

        # explain all the predictions in the test set
        background = shap.kmeans(self.train_x, 10)
        explainer = shap.KernelExplainer(model.predict, background)
        background = self.test_x[np.random.choice(self.test_x.shape[0], 10000, replace=False)]
        shap_values = explainer.shap_values(background)
        plt.figure()
        shap.summary_plot(shap_values, background, feature_names, show=False)
        # shap.plots.beeswarm(shap_vals, feature_names)
        # shap.plots.waterfall(shap_values[1], max_display=10)
        plt.savefig(
            out_dir + "/Neural_network_features.pdf",
            format="pdf",
            dpi=1000,
            bbox_inches="tight",
        )
        del background, shap_values, model, study
        return None


def data_parsing(train_x, train_y, test_x, test_y, config_dict):
    # Load data
    # print(f'\nUsing merged_data-train_{var}..', file=open(output, 'a'))
    X_train = pd.read_csv(train_x)
    # var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict["id_cols"], axis=1)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    feature_names = X_train.columns.tolist()
    X_train = X_train.values
    Y_train = pd.read_csv(train_y)
    Y_train = label_binarize(
        Y_train.values, classes=list(np.unique(Y_train))
    ).ravel()
    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(Y_train),y=Y_train)
    class_weights = {i:w for i,w in enumerate(class_weights)}
    #Y_train = pd.get_dummies(Y_train)


    X_test = pd.read_csv(test_x)
    # var = X_train[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict["id_cols"], axis=1)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)
    X_test = X_test.values
    Y_test = pd.read_csv(test_y)
    #Y_test = pd.get_dummies(Y_test)
    Y_test = label_binarize(
        Y_test.values, classes=list(np.unique(Y_test))
    ).ravel()

    print(f"Shape: {Y_train.shape}")
    print("Data Loaded!")
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test, feature_names, class_weights


def is_valid_file(p, arg):
    if not Path(os.path.expandvars(arg)).is_file():
        p.error(f"The file {arg} does not exist!")
    else:
        return os.path.expandvars(arg)


def is_valid_dir(p, arg):
    if not Path(os.path.expandvars(arg)).is_dir():
        p.error(f"The directory {arg} does not exist!")
    else:
        return os.path.expandvars(arg)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Script to train and tune DITTO using processed annotations from OpenCravat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    PARSER.add_argument(
        "--train_x",
        help="File path to the CSV file of X_train data",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )
    PARSER.add_argument(
        "--train_y",
        help="File path to the CSV file of y_train data",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )
    PARSER.add_argument(
        "--test_x",
        help="File path to the CSV file of X_test data",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )
    PARSER.add_argument(
        "--test_y",
        help="File path to the CSV file of y_test data",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )
    PARSER.add_argument(
        "-c",
        "--config",
        help="File path to the data config JSON file that determines how to process annotated variants from OpenCravat",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )
    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--outdir",
        help="Output directory to save files from training/tuning DITTO",
        type=lambda x: is_valid_dir(PARSER, x),
        metavar="\b",
    )

    ARGS = PARSER.parse_args()
    out_dir = ARGS.outdir if ARGS.outdir else f"{Path().resolve()}"

    with open(ARGS.config) as fh:
        config_dict = yaml.safe_load(fh)

    start = time.perf_counter()

    output = out_dir + "/ML_results.csv"
    # print('Working with '+var+' dataset...', file=open(output, "w"))
    print("Working with dataset...")

    X_train, X_test, Y_train, Y_test, feature_names, class_weights = data_parsing(
        ARGS.train_x, ARGS.train_y, ARGS.test_x, ARGS.test_y, config_dict
    )

    print("Starting Objective...")
    objective = Objective(X_train, X_test, Y_train, Y_test, class_weights)
    tensorboard_callback = TensorBoardCallback(
        out_dir + "/Neural_network_logs/", metric_name="accuracy"
    )
    study = optuna.create_study(
        sampler=TPESampler(**TPESampler.hyperopt_parameters()),
        study_name=f"DITTO_NN",
        storage=f"sqlite:///{out_dir}/Neural_network.db",  # study_name= "Ditto3",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2),
        load_if_exists=True,  # , pruner=optuna.pruners.MedianPruner(n_startup_trials=150)
    )
    # study = optuna.load_study(study_name= "Ditto_all", sampler=TPESampler(**TPESampler.hyperopt_parameters()),storage ="sqlite:///Ditto_all.db") # study_name= "Ditto3",
    study.optimize(
        objective,
        n_trials=5,
        callbacks=[tensorboard_callback],
        n_jobs=-1,
        gc_after_trial=True,
    )
    finish = (time.perf_counter() - start) / 120
    print(f"Total time in hrs: {finish}")
    objective.show_result(study, out_dir, output, feature_names)
    del X_train, X_test, Y_train, Y_test, feature_names
