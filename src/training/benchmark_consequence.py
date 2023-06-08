import pandas as pd
pd.set_option('display.max_rows', None)
import warnings
warnings.simplefilter("ignore")
import yaml
import argparse
import shap
import numpy as np
import matplotlib.pyplot as plt
import functools
print = functools.partial(print, flush=True)
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)


def get_roc_plot(so, X_test, Y_test, outdir):

    # prepare plots
    fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(50, 20))
    fig.suptitle(f"DITTO Benchmarking for {so}", fontsize=40)
    fsize = 30

    ax_roc.set_xlabel("False Positive Rate", fontsize=fsize)
    ax_roc.set_ylabel("True Positive Rate", fontsize=fsize)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves", fontsize=fsize)
    ax_roc.grid(linestyle="--")
    ax_prc.set_xlabel("Recall", fontsize=fsize)
    ax_prc.set_ylabel("Precision", fontsize=fsize)
    ax_prc.set_title("Precision Recall (PRC) curves", fontsize=fsize)
    ax_prc.grid(linestyle="--")

    for name in list(X_test.columns):
            fpr, tpr, _ = roc_curve(Y_test, X_test[name].fillna(0).values)
            try:
                auc = roc_auc_score(Y_test, X_test[name].fillna(0).values)
            except:
                auc=np.nan
            auc = "{:.2f}".format(auc)
            ax_roc.plot(fpr,tpr,label=str(name)+" = "+str(auc))
            precision, recall, _ = precision_recall_curve(Y_test, X_test[name].fillna(0).values)
            try:
                prc = average_precision_score(Y_test, X_test[name].fillna(0).values)
            except:
                prc=np.nan
            prc = "{:.2f}".format(prc)
            ax_prc.plot(recall,precision,label=str(name)+" = "+str(prc))


    #ax_prc.legend(bbox_to_anchor=(1,0), loc="lower left", fontsize=20)
    ax_prc.legend( loc="lower right", fontsize=20)
    ax_roc.legend( loc="lower right", fontsize=20)
    fig.savefig(
        f"{outdir}/{so}_ROC_PRC_benchmarking.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    return None

def get_prediction(clf, X_test):
    y_score = clf.predict(X_test.values)
    return y_score

def get_SHAP(test_x, clf, so, outdir, feature_names):
    if test_x.shape[0] > 100:
        sample_size = 100
    else:
        sample_size = test_x.shape[0]
    background_x = test_x[np.random.choice(test_x.shape[0], sample_size, replace=False)]
    explainer = shap.KernelExplainer(model = clf, data = background_x) #(background_so, background_x))
    shap_values = explainer.shap_values(background_x)

    plt.clf()
    plt.xlabel("mean SHAP value")
    plt.ylabel("Features")
    plt.title(f"SHAP plot for {so}")
    shap.summary_plot(shap_values, background_x, feature_names, show=False)
    #plt.show()
    plt.savefig(
            f"{outdir}/{so}_SHAP.pdf",
            format="pdf",
            dpi=1000,
            bbox_inches="tight",
        )
    return None

def run_test(X_test, outdir, feature_names, clf, config_dict):
    consequence_list = list(set(config_dict['consequence_cols']) & set(X_test.columns))
    for so in consequence_list:
        test_x = X_test[X_test[so]==1]
        try:
            benchmark_df = test_x[config_dict['Benchmark_cols'].keys()]
            Y_test = test_x['class']
            test_x = test_x.drop(["class"], axis=1)
            test_x.rename(columns=config_dict['Benchmark_cols'], inplace=True)
            y_score = get_prediction(clf, test_x)
            benchmark_df['DITTO'] = y_score
            get_roc_plot(so, benchmark_df, Y_test, outdir)
        except:
            print(f"Error occured for {so} ROC plot!")
            pass

def data_parsing(test_x, test_y, config_dict):
    # Load data
    X_test = pd.read_csv(test_x)
    X_test = X_test.drop(config_dict["id_cols"], axis=1)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(0, inplace=True)
    feature_names = X_test.columns.tolist()
    #X_test = X_test.values
    Y_test = pd.read_csv(test_y)
    #Y_test = pd.get_dummies(Y_test)
    Y_test = label_binarize(
        Y_test.values, classes=list(np.unique(Y_test))
    ).ravel().reshape(-1, 1)
    X_test['class'] = Y_test

    print(f"Shape: {Y_test.shape}")
    #print(X_test['class'])
    print("Data Loaded!")
    # scaler = StandardScaler().fit(X_train)
    # X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)
    return X_test,feature_names

def load_model(model_path):
    clf = keras.models.load_model(model_path + '/Neural_network')
    clf.load_weights(model_path + "/weights.h5")
    return clf

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
        description="Script to benchmark DITTO using processed annotations from OpenCravat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    PARSER.add_argument(
        "-d",
        "--ditto",
        help="Folder path to the DITTO model",
        required=True,
        type=lambda x: is_valid_dir(PARSER, x),
        metavar="\b",
    )
    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--outdir",
        help="Output directory to save files from DITTO",
        type=lambda x: is_valid_dir(PARSER, x),
        metavar="\b",
    )

    ARGS = PARSER.parse_args()
    out_dir = ARGS.outdir if ARGS.outdir else f"{Path().resolve()}"

    with open(ARGS.config) as fh:
        config_dict = yaml.safe_load(fh)

    clf = load_model(
        ARGS.ditto
    )
    X_test, feature_names = data_parsing(
        ARGS.test_x, ARGS.test_y, config_dict
    )

    run_test(X_test, out_dir, feature_names, clf, config_dict)
