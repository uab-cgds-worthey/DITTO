# python slurm-launch.py --exp-name Learning_curve --command "python training/training/stacking_LC_error.py" --partition short --mem 50G

# from numpy import mean
import numpy as np
import pandas as pd
import time
import argparse
import ray

# Start Ray.
ray.init(ignore_reinit_error=True)
import warnings

warnings.simplefilter("ignore")

from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import functools

print = functools.partial(print, flush=True)
import gc
import os

os.chdir(
    "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
)

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

def data_parsing(var, config_dict):
    # Load data
    # print(f'\nUsing merged_data-train_{var}..', file=open(output, "a"))
    X_train = pd.read_csv(f"train_custom_data-{var}.csv")
    # var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict["var"], axis=1)
    #X_train = X_train.sample(frac=1).reset_index(drop=True)
    X_train = X_train.values
    Y_train = pd.read_csv(f"train_custom_data-y-{var}.csv")
    Y_train = label_binarize(
        Y_train.values, classes=["low_impact", "high_impact"]
    ).ravel()


    return X_train, Y_train



# @ray.remote #(num_cpus=9)
def classifier(
    clf, var, X, y
):
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("Learning Curves")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        clf,
        X,
        y,
        cv=5,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        return_times=True,
        train_sizes=np.linspace(0.001, 1.0, 10),
    )
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].plot(train_sizes, -train_scores.mean(1), "o--", color="r", label="Training score")
    axes[0].plot(train_sizes, -test_scores.mean(1), "o--", color="g", label="Cross-validation score")
    axes[0].set_xlabel("Train size")
    axes[0].set_ylabel("Neg Mean Squared Error")
    axes[0].set_title("Learning curves")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Neg Mean Squared Error")
    axes[2].set_title("Performance of the model")
    plt.savefig( "./models_custom/" + var + "/Stacking_LC_error_" + var +".png")
    plt.close()

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        #required=True,
        default="dbnsfp",
        help="The tag used when generating train/test data. Default:'dbnsfp'",
    )

    args = parser.parse_args()

    # Classifiers I wish to use
    classifiers = StackingClassifier(
        estimators=[
            #("DecisionTree", DecisionTreeClassifier(class_weight="balanced")),
            #(
            #    "RandomForest",
            #    RandomForestClassifier(class_weight="balanced", n_jobs=-1),
            #),
            ("BalancedRF", BalancedRandomForestClassifier()),
            #("AdaBoost", AdaBoostClassifier()),
            ("ExtraTrees", ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)),
            #("GaussianNB", GaussianNB()),
             ("LDA", LinearDiscriminantAnalysis()),
            ("GradientBoost", GradientBoostingClassifier()),
             ("MLP", MLPClassifier())
        ],
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
        final_estimator=LogisticRegression(n_jobs=-1),
        verbose=1,
    )

    with open("../../configs/col_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    var = args.var_tag
    if not os.path.exists("./models_custom/" + var):
        os.makedirs("./models_custom/" + var)
    # print('Working with '+var+' dataset...', file=open(output, "a"))
    print("Working with " + var + " dataset...")
    X_train,Y_train = data_parsing(
        var, config_dict
    )

    classifier(
        classifiers,
        var,
        X_train,
        Y_train,
    )
    gc.collect()
