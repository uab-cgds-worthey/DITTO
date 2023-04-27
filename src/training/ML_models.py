# python slurm-launch.py --exp-name Training --command "python training/ML_models.py" --partition largemem --mem 500G

import numpy as np
import pandas as pd
import time
import argparse
import ray

# Start Ray.
ray.init(ignore_reinit_error=True)
import warnings

warnings.simplefilter("ignore")
from joblib import dump, load
import shap

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score,
    recall_score,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import gc
import os

os.chdir(
    "/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/"
)

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

# @ray.remote(num_returns=6)
def data_parsing(config_dict):
    # Load data
    # print(f'\nUsing merged_data-train_{var}..', file=open(output, "a"))
    X_train = pd.read_csv(f"train_data_80.csv.gz")
    X_val = pd.read_csv(f"val_data_20.csv.gz")

    # concatenating df1 and df2 along rows
    X_train = pd.concat([X_train, X_val], axis=0)
    del X_val
    # var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict["id_cols"], axis=1)
    feature_names = X_train.columns.tolist()
    #X_train = X_train.sample(frac=1).reset_index(drop=True)
    X_train = X_train.values
    Y_train = pd.read_csv(f"train_data-y_80.csv.gz")
    Y_val = pd.read_csv(f"val_data-y_20.csv.gz")
    Y_train = pd.concat([Y_train, Y_val], axis=0)
    del Y_val
    Y_train = label_binarize(
        Y_train.values, classes=["low_impact", "high_impact"]
    ).ravel()

    X_test = pd.read_csv(f"test_data_20.csv.gz")
    # var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict["id_cols"], axis=1)
    # feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f"test_data-y_20.csv.gz")
    print("Data Loaded!")
    # Y = pd.get_dummies(y)
    Y_test = label_binarize(
        Y_test.values, classes=["low_impact", "high_impact"]
    ).ravel()

    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    # explain all the predictions in the test set
    background = shap.kmeans(X_train, 10)
    return X_train, X_test, Y_train, Y_test, background, feature_names


@ray.remote  # (num_cpus=9)
def classifier(
    name, clf, X_train, X_test, Y_train, Y_test, background, feature_names, output
):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/DITTO/data/processed/"
    )
    start = time.perf_counter()
    score = cross_validate(
        clf,
        X_train,
        Y_train,
        cv=10,
        return_train_score=True,
        return_estimator=True,
        n_jobs=-1,
        verbose=0,
        scoring=("roc_auc", "neg_log_loss"),
    )
    clf = score["estimator"][np.argmin(score["test_neg_log_loss"])]
    # y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    # clf.fit(X_train, Y_train) #, class_weight=class_weights)
    # name = str(type(clf)).split("'")[1]  #.split(".")[3]
    with open(f"./models/{name}.joblib", "wb") as f:
        dump(clf, f, compress="lz4")
    # del clf
    # with open(f"./models/{var}/{name}_{var}.joblib", 'rb') as f:
    # clf = load(f)
    y_score = clf.predict(X_test)
    prc = precision_score(Y_test, y_score, average="weighted")
    recall = recall_score(Y_test, y_score, average="weighted")
    roc_auc = roc_auc_score(Y_test, y_score)
    prc_auc = average_precision_score(Y_test, y_score, average="weighted")
    # roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
    accuracy = accuracy_score(Y_test, y_score)
    # score = clf.score(X_train, Y_train)
    # matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
    matrix = confusion_matrix(Y_test, y_score)

    finish = (time.perf_counter() - start) / 60
    with open(output, "a") as f:
        f.write(
            #f"{name}\t{np.mean(score['train_roc_auc'])}\t{np.mean(score['test_roc_auc'])}\t{np.mean(score['train_neg_log_loss'])}\t{np.mean(score['test_neg_log_loss'])}\t{prc}\t{recall}\t{roc_auc}\t{prc_auc}\t{accuracy}\t{finish}\n{matrix}\n"
            f"{name}\t{np.mean(score['train_roc_auc'])}\t{np.mean(score['test_roc_auc'])}\t{np.mean(score['train_neg_log_loss'])}\t{np.mean(score['test_neg_log_loss'])}\t{prc}\t{recall}\t{roc_auc}\t{prc_auc}\t{accuracy}\t{finish}\n"
        )
    # explain all the predictions in the test set
    # background = shap.kmeans(X_train, 6)
    explainer = shap.KernelExplainer(clf.predict, background)
    del clf, X_train
    background1 = X_test[np.random.choice(X_test.shape[0], 5000, replace=False)]
    shap_values = explainer.shap_values(background1)
    plt.figure()
    shap.summary_plot(shap_values, background1, feature_names, max_display = 30, show=False)
    # shap.plots.waterfall(shap_values[0], max_display=15)
    plt.savefig(
        f"./models/{name}_features.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    del shap_values, background1, explainer
    return None


if __name__ == "__main__":

    # Classifiers I wish to use
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(class_weight="balanced"),
        "RandomForest": RandomForestClassifier(class_weight="balanced", n_jobs=5),
        "BalancedRF": BalancedRandomForestClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(class_weight="balanced", n_jobs=5),
        "GaussianNB": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "GradientBoost": GradientBoostingClassifier(),
        "MLP": MLPClassifier(),
    }

    with open("../../configs/col_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)


    if not os.path.exists("./models/" ):
        os.makedirs("./models/")
    output = "./models" + "/ML_results.csv"
    # print('Working with '+var+' dataset...', file=open(output, "a"))
    print("Working with dataset...")
    X_train, X_test, Y_train, Y_test, background, feature_names = data_parsing(
        config_dict
    )
    with open(output, "w") as f:
        f.write(
            #"Model\tCross_validate(avg_train_roc_auc)\tCross_validate(avg_test_roc_auc)\tCross_validate(avg_train_neg_log_loss)\tCross_validate(avg_test_neg_log_loss)\tPrecision(test_data)\tRecall\troc_auc\tprc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]\n"
            "Model\tCross_validate(avg_train_roc_auc)\tCross_validate(avg_test_roc_auc)\tCross_validate(avg_train_neg_log_loss)\tCross_validate(avg_test_neg_log_loss)\tPrecision(test_data)\tRecall\troc_auc\tprc_auc\tAccuracy\tTime(min)\n"
        )
    remote_ml = [
        classifier.remote(
            name,
            clf,
            X_train,
            X_test,
            Y_train,
            Y_test,
            background,
            feature_names,
            output,
        )
        for name, clf in classifiers.items()
    ]
    ray.get(remote_ml)
    gc.collect()

    # prepare plots
    fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Model performances on Testing data with filters", fontsize=20)

    for name, clf in classifiers.items():

        with open(f"./models/{name}.joblib", "rb") as f:
            clf = load(f)

        plot_precision_recall_curve(clf, X_test, Y_test, ax=ax_prc, name=name)
        plot_roc_curve(clf, X_test, Y_test, ax=ax_roc, name=name)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_prc.set_title("Precision Recall (PRC) curves")

    ax_roc.grid(linestyle="--")
    ax_prc.grid(linestyle="--")

    plt.legend()
    plt.savefig(
        f"./models/roc.pdf", format="pdf", dpi=1000, bbox_inches="tight"
    )
    gc.collect()
