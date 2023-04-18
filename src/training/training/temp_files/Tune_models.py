# from numpy import mean
import numpy as np
import pandas as pd
import time
import ray
from ray import tune
from tune_sklearn import TuneSearchCV

# Start Ray.
ray.init(ignore_reinit_error=True)
import warnings

warnings.simplefilter("ignore")
import joblib
from joblib import dump, load
import shap

# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE
# from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    precision_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    recall_score,
)

# from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
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
from functools import partial

print = partial(print, flush=True)
import os
from ray.util.multiprocessing import Pool

print(
    f"""This cluster consists of
    {len(ray.nodes())} nodes in total
    {ray.cluster_resources()['CPU']} CPU resources in total
"""
)


@ray.remote
def data_parsing(var, config_dict, output):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )
    # Load data
    print(f"\nUsing merged_data-train_{var}..", file=open(output, "a"))
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


@ray.remote
def classifier(clf, model, var, X_train, X_test, Y_train, Y_test, feature_names):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )
    start = time.perf_counter()
    # score = cross_validate(clf, X_train, Y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=0, scoring=('roc_auc','neg_log_loss'))
    # clf = score['estimator'][np.argmin(score['test_neg_log_loss'])]
    # y_score = cross_val_predict(clf, X_train, Y_train, cv=5, n_jobs=-1, verbose=0)
    # class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    # clf.fit(X_train, Y_train) #, class_weight=class_weights)
    # clf_name = str(type(clf)).split("'")[1]  #.split(".")[3]
    with open(f"./tuning/{var}/{model}_{var}.joblib", "wb") as f:
        dump(clf, f, compress="lz4")
    # del clf
    # with open(f"./models/{var}/{clf_name}_{var}.joblib", 'rb') as f:
    # clf = load(f)
    y_score = clf.predict(X_test)
    prc = precision_score(Y_test, y_score, average="weighted")
    recall = recall_score(Y_test, y_score, average="weighted")
    roc_auc = roc_auc_score(Y_test, y_score)
    # roc_auc = roc_auc_score(Y_test, np.argmax(y_score, axis=1))
    accuracy = accuracy_score(Y_test, y_score)
    score = clf.score(X_train, Y_train)
    # matrix = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(y_score, axis=1))
    matrix = confusion_matrix(Y_test, y_score)

    # explain all the predictions in the test set
    # background = shap.kmeans(X_train, 6)
    # explainer = shap.KernelExplainer(clf.predict, background)
    del clf, X_train
    # background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
    # shap_values = explainer.shap_values(background)
    # plt.figure()
    # shap.summary_plot(shap_values, background, feature_names, show=False)
    ##shap.plots.waterfall(shap_values[0], max_display=15)
    # plt.savefig(f"./models/{var}/{clf_name}_{var}_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    finish = (time.perf_counter() - start) / 60
    list1 = [model, prc, recall, roc_auc, accuracy, score, finish, matrix]
    # list1 = [clf_name, np.mean(score['train_roc_auc']), np.mean(score['test_roc_auc']),np.mean(score['train_neg_log_loss']), np.mean(score['test_neg_log_loss']), prc, recall, roc_auc, accuracy, finish, matrix]
    # pickle.dump(clf, open("./models/"+var+"/"+clf_name+"_"+var+".pkl", 'wb'))
    return list1


def tuning(var, X_train, X_test, Y_train, Y_test, feature_names, output, models):
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )
    for model, config in models.items():
        # return model, config, len(X_train), len(Y_train)
        hyperopt_tune_search = TuneSearchCV(
            model,
            param_distributions=config,
            n_trials=10,
            early_stopping=False,
            max_iters=1,  # max_iters specifies how many times tune-sklearn will be given the decision to start/stop training a model. Thus, if you have early_stopping=False, you should set max_iters=1 (let sklearn fit the entire estimator).
            search_optimization="bayesian",
            n_jobs=-1,
            refit=True,
            cv=5,
            verbose=1,
            # loggers = "tensorboard",
            random_state=42,
            local_dir="./ray_results",
        )
        hyperopt_tune_search.fit(X_train, Y_train)
        best_model = hyperopt_tune_search.best_estimator_
        print(
            f"{model}_{var}:{hyperopt_tune_search.best_params_}",
            file=open("tuning/tuned_parameters.csv", "a"),
        )
        list1 = ray.get(
            classifier.remote(
                best_model, model, var, X_train, X_test, Y_train, Y_test, feature_names
            )
        )
        # print(f'{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\t{list1[7]}\t{list1[8]}\t{list1[9]}\n{list1[10]}', file=open(output, "a"))
        print(
            f"{list1[0]}\t{list1[1]}\t{list1[2]}\t{list1[3]}\t{list1[4]}\t{list1[5]}\t{list1[6]}\n{list1[7]}",
            file=open(output, "a"),
        )
        del best_config, best_model
        return model


if __name__ == "__main__":

    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/"
    )

    # Classifiers I wish to use
    classifiers = [
        {
            ExtraTreesClassifier(): {  # bootstrap = True,
                #  warm_start=True,
                #  oob_score=True): {
                "n_estimators": tune.randint(0, 200),
                "min_samples_split": tune.randint(2, 100),
                "min_samples_leaf": tune.randint(1, 100),
                "criterion": tune.choice(["gini", "entropy"]),
                "max_features": tune.choice(["sqrt", "log2"]),
                # "oob_score" : tune.choice([True, False]),
                "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
            }
        },
        {
            DecisionTreeClassifier(): {
                "min_samples_split": tune.randint(2, 100),
                "min_samples_leaf": tune.randint(1, 100),
                "criterion": tune.choice(["gini", "entropy"]),
                "max_features": tune.choice(["sqrt", "log2"]),
                "class_weight": tune.choice([None, "balanced"]),
            }
        },
        {
            SGDClassifier(n_jobs=-1): {
                "loss": tune.choice(
                    [
                        "squared_hinge",
                        "hinge",
                        "log",
                        "modified_huber",
                        "perceptron",
                        "squared_loss",
                        "huber",
                        "epsilon_insensitive",
                        "squared_epsilon_insensitive",
                    ]
                ),
                "penalty": tune.choice(["l2", "l1", "elasticnet"]),
                "alpha": tune.loguniform(1e-9, 1e-1),
                "epsilon": tune.uniform(1e-9, 1e-1),
                "fit_intercept": tune.choice([True, False]),
                "learning_rate": tune.choice(
                    ["constant", "optimal", "invscaling", "adaptive"]
                ),
                "class_weight": tune.choice([None, "balanced"]),
            }
        },
        {
            RandomForestClassifier(n_jobs=-1): {
                "n_estimators": tune.randint(0, 200),
                "min_samples_split": tune.randint(2, 100),
                "min_samples_leaf": tune.randint(1, 100),
                "criterion": tune.choice(["gini", "entropy"]),
                "max_features": tune.choice(["sqrt", "log2"]),
                "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
                # "oob_score" : tune.choice([True, False]),
                "max_depth": tune.randint(0, 200),
            }
        },
        {
            AdaBoostClassifier(): {
                "n_estimators": tune.randint(0, 200),
                "algorithm": tune.choice(["SAMME", "SAMME.R"]),
            }
        },
        {
            BalancedRandomForestClassifier(): {
                "n_estimators": tune.randint(0, 200),
                "min_samples_split": tune.randint(2, 100),
                "min_samples_leaf": tune.randint(1, 100),
                "criterion": tune.choice(["gini", "entropy"]),
                "max_features": tune.choice(["sqrt", "log2"]),
                "class_weight": tune.choice([None, "balanced", "balanced_subsample"]),
                # "oob_score" : tune.choice([True, False]),
                "max_depth": tune.randint(0, 200),
            }
        },
        # {GaussianNB(): {
        #    #"var_smoothing": tune.loguniform(0.01, 1.0),
        # }},
        {
            LinearDiscriminantAnalysis(): {
                "solver": tune.choice(["svd", "lsqr", "eigen"])
            }
        },
        {
            GradientBoostingClassifier(): {
                "n_estimators": tune.randint(0, 200),
                "min_samples_split": tune.randint(2, 100),
                "min_samples_leaf": tune.randint(1, 100),
                "max_features": tune.choice(["sqrt", "log2"]),
                "max_features": tune.randint(1, 10),
                "subsample": tune.uniform(0.0, 1.0),
                "learning_rate": tune.loguniform(0.01, 1.0),
                "max_depth": tune.randint(2, 200),
            }
        },
        # {MLPClassifier(): {
        #    "hidden_layer_sizes": tune.sample_from(lambda _: [tune.randint(1, 100) for i in range(tune.randint(1, 50))]),
        #    "activation": tune.choice(['identity', 'logistic', 'tanh', 'relu']),
        #    "solver": tune.choice(['lbfgs', 'sgd', 'adam']),
        #    'alpha': tune.loguniform(1e-9, 1e-1),
        #    'learning_rate': tune.choice(['constant','adaptive','invscaling']),
        #    'tol': tune.loguniform(1e-9, 1e-1),
        #    'epsilon': tune.uniform(1e-9, 1e-1),
        #    "max_iter": tune.randint(10, 300),
        # }},
    ]

    with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    ##variants = ['snv','non_snv','snv_protein_coding'] #'snv',
    variants = ["non_snv"]
    for var in variants:
        if not os.path.exists("tuning/" + var):
            os.makedirs("./tuning/" + var)
        output = "tuning/" + var + "/ML_results_" + var + ".csv"
        print("Working with " + var + " dataset...", file=open(output, "w"))
        print("Working with " + var + " dataset...")
        X_train, X_test, Y_train, Y_test, feature_names = ray.get(
            data_parsing.remote(var, config_dict, output)
        )
        #    #print('Model\tCross_validate(avg_train_roc_auc)\tCross_validate(avg_test_roc_auc)\tCross_validate(avg_train_neg_log_loss)\tCross_validate(avg_test_neg_log_loss)\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tTime(min)\tConfusion_matrix[low_impact, high_impact]', file=open(output, "a"))    #\tConfusion_matrix[low_impact, high_impact]
        print(
            "Model\tPrecision(test_data)\tRecall\troc_auc\tAccuracy\tScore\tTime(min)\tConfusion_matrix[low_impact, high_impact]",
            file=open(output, "a"),
        )  # \tConfusion_matrix[low_impact, high_impact]
        pool = Pool(ray_address="auto")
        func = partial(
            tuning, var, X_train, X_test, Y_train, Y_test, feature_names, output
        )
        for models in pool.map(func, classifiers):
            # for model, config in zip(classifiers.keys(), classifiers.values()):
            # best_config, hyperopt_tune_search = ray.get(tuning.remote(model,config, X_train, Y_train))
            # print(models[0], models[1], models[2], models[3])
            # best_config, best_model, model= models[0], models[1], models[2]
            print(f"{models} training and testing done!")
