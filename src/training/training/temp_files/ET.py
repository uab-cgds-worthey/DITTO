# from numpy import mean
import numpy as np
import pandas as pd
import time
import ray
from ray import tune
import argparse
from tune_sklearn import TuneSearchCV
import warnings

warnings.simplefilter("ignore")
from joblib import dump, load
import shap

# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
)

# from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import yaml
import functools

print = functools.partial(print, flush=True)
import os
import gc


def data_parsing(var, config_dict, output):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )
    # Load data
    print(f"\nUsing merged_data-train_{var}..")
    X_train = pd.read_csv(f"train_{var}/merged_data-train_{var}.csv")
    # var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict["ML_VAR"], axis=1)
    feature_names = X_train.columns.tolist()
    X_train = X_train.values
    Y_train = pd.read_csv(f"train_{var}/merged_data-y-train_{var}.csv")
    Y_train = label_binarize(
        Y_train.values, classes=["low_impact", "high_impact"]
    ).ravel()

    X_test = pd.read_csv(f"test_{var}/merged_data-test_{var}.csv")
    # var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict["ML_VAR"], axis=1)
    # feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f"test_{var}/merged_data-y-test_{var}.csv")
    print("Data Loaded!")
    # Y = pd.get_dummies(y)
    Y_test = label_binarize(
        Y_test.values, classes=["low_impact", "high_impact"]
    ).ravel()

    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    # explain all the predictions in the test set
    # background = shap.kmeans(X_train, 10)
    return X_train, X_test, Y_train, Y_test, feature_names


def tuning(var, X_train, X_test, Y_train, Y_test, feature_names, output):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )
    model = ExtraTreesClassifier()
    config = {  # bootstrap = True,
        #  warm_start=True,
        #  oob_score=True): {
        "n_estimators": tune.randint(1, 200),
        "min_samples_split": tune.randint(2, 100),
        "min_samples_leaf": tune.randint(1, 100),
        "criterion": tune.choice(["gini", "entropy"]),
        "max_features": tune.choice(["sqrt", "log2"]),
        # "oob_score" : tune.choice([True, False]),
        "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
    }
    start = time.perf_counter()
    clf = TuneSearchCV(
        model,
        param_distributions=config,
        n_trials=300,
        early_stopping=False,
        max_iters=1,  # max_iters specifies how many times tune-sklearn will be given the decision to start/stop training a model. Thus, if you have early_stopping=False, you should set max_iters=1 (let sklearn fit the entire estimator).
        search_optimization="bayesian",
        n_jobs=30,
        refit=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        verbose=0,
        # loggers = "tensorboard",
        random_state=42,
        local_dir="./ray_results",
    )
    clf.fit(X_train, Y_train)
    print(
        f"{model}_{var}:{clf.best_params_}",
        file=open("tuning/tuned_parameters.csv", "a"),
    )
    clf = clf.best_estimator_

    score = clf.score(X_train, Y_train)
    clf_name = str(type(model)).split("'")[1]  # .split(".")[3]
    with open(f"./tuning/{var}/{model}_{var}.joblib", "wb") as f:
        dump(clf, f, compress="lz4")
    # with open(f"./models/{var}/{clf_name}_{var}.joblib", 'rb') as f:
    # clf = load(f)
    y_score = clf.predict(X_test)
    prc = precision_score(Y_test, y_score, average="weighted")
    recall = recall_score(Y_test, y_score, average="weighted")
    roc_auc = roc_auc_score(Y_test, y_score)
    accuracy = accuracy_score(Y_test, y_score)
    matrix = confusion_matrix(Y_test, y_score)
    finish = (time.perf_counter() - start) / 60
    print(
        "Model\tScore\tPrecision\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]",
        file=open(output, "a"),
    )  # \tConfusion_matrix[low_impact, high_impact]
    print(
        f"{clf_name}\t{score}\t{prc}\t{recall}\t{roc_auc}\t{accuracy}\t{finish}\n{matrix}",
        file=open(output, "a"),
    )
    del Y_test
    # explain all the predictions in the test set
    background = shap.kmeans(X_train, 10)
    del X_train, Y_train
    explainer = shap.KernelExplainer(clf.predict, background)
    del clf, background
    background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
    del X_test
    shap_values = explainer.shap_values(background)
    plt.figure()
    shap.summary_plot(shap_values, background, feature_names, show=False)
    plt.savefig(
        f"./tuning/{var}/{clf_name}_{var}_features.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    del background, explainer, feature_names
    print(f"{model} training and testing done!")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--vtype",
        type=str,
        default="non_snv",
        help="Type of variation/s (without spaces between) to tune the classifier (like: snv,non_snv,snv_protein_coding). (Default: non_snv)",
    )

    args = parser.parse_args()

    var = args.vtype

    if args.smoke_test:
        ray.init(num_cpus=2)  # force pausing to happen for test
    else:
        ray.init(ignore_reinit_error=True)

    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )

    with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    ##variants = ['snv','non_snv','snv_protein_coding']

    if not os.path.exists("tuning/" + var):
        os.makedirs("./tuning/" + var)
    output = "tuning/" + var + "/ML_results_" + var + ".csv"
    # print('Working with '+var+' dataset...', file=open(output, "w"))
    print("Working with " + var + " dataset...")
    X_train, X_test, Y_train, Y_test, feature_names = data_parsing(
        var, config_dict, output
    )
    tuning(var, X_train, X_test, Y_train, Y_test, feature_names, output)
    gc.collect()
