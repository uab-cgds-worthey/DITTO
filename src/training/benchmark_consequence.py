# This script is used to benchmark DITTO. This script outputs roc/prc plots, confusion matrices, and SHAP plots for each
# consequence. This is replacement for the script `src/analysis/opencravat_latest_benchmarking-Consequence_80_20.ipynb`.

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
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
)
from sklearn.preprocessing import label_binarize, MinMaxScaler
from sklearn.utils import class_weight
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
tf.config.run_functions_eagerly(True)


def get_roc_plot(so, X_test, Y_test, outdir, weights):

    # prepare plots
    fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(50, 20))
    fig.suptitle(f"DITTO Benchmarking for {so}", fontsize=40)
    fsize = 30
    ax_roc.tick_params(axis='both', which='major', labelsize=fsize)
    ax_prc.tick_params(axis='both', which='major', labelsize=fsize)
    ax_roc.set_xlabel("False Positive Rate", fontsize=fsize)
    ax_roc.set_ylabel("True Positive Rate", fontsize=fsize)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves", fontsize=fsize)
    ax_roc.grid(linestyle="--")
    ax_prc.set_xlabel("Recall", fontsize=fsize)
    ax_prc.set_ylabel("Precision", fontsize=fsize)
    ax_prc.set_title("Precision Recall (PRC) curves", fontsize=fsize)
    ax_prc.grid(linestyle="--")

    roc_scores_so = {}
    prc_scores_so = {}
    f1_scores_so = {}
    for name in list(X_test.columns):
            fpr, tpr, _ = roc_curve(Y_test, X_test[name].round(), sample_weight= weights)
            auc = roc_auc_score(Y_test, X_test[name].round(), sample_weight= weights, average='weighted')
            auc = "{:.2f}".format(auc)
            roc_scores_so[name] = auc
            ax_roc.plot(fpr,tpr,label=str(name)+" = "+str(auc))
            precision, recall, _ = precision_recall_curve(Y_test, X_test[name].round(), sample_weight= weights)
            prc = average_precision_score(Y_test, X_test[name].round(), sample_weight= weights, average='weighted')
            prc = "{:.2f}".format(prc)
            prc_scores_so[name] = prc
            f1 = f1_score(Y_test, X_test[name].round(), sample_weight= weights, average='weighted')
            f1_scores_so[name] = "{:.2f}".format(f1)
            ax_prc.plot(recall,precision,label=str(name)+" = "+str(prc))
    ax_prc.legend( bbox_to_anchor=(1,0.5), loc="center left", fontsize=fsize)
    ax_roc.legend( bbox_to_anchor=(1,0.5), loc="center left", fontsize=fsize)
    fig.tight_layout()

    fig.savefig(
        f"{outdir}/{so}_ROC_PRC_benchmarking.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    return roc_scores_so, prc_scores_so, f1_scores_so

def get_prediction(clf, X_test):
    y_score = clf.predict(X_test.values)
    return y_score

def get_matrix(y_score, Y_test, so, outdir):
    cm = confusion_matrix(Y_test,y_score.round())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Path', 'Benign'])
    disp.plot()
    plt.title(f"Confusion matrix for {so}", fontsize=15)
    plt.savefig(
        f"{outdir}/{so}_matrix.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )
    return None

def get_SHAP(test_x, clf, so, outdir, feature_names):
    if test_x.shape[0] > 500:
        sample_size = 500
    else:
        sample_size = test_x.shape[0]
    background_x = test_x.sample(n=sample_size)
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

def run_test(X_test, outdir, clf, config_dict, feature_names, class_weights):
    consequence_list = list(set(config_dict['consequence_cols']) & set(X_test.columns))
    roc_scores = {}
    prc_scores = {}
    f1_scores = {}
    for so in consequence_list:
        roc_scores[so] = {}
        prc_scores[so] = {}
        f1_scores[so] = {}
        test_x = X_test[X_test[so]==1]
        Y_test = test_x['class']
        test_x = test_x.drop(["class"], axis=1)
        if not test_x.empty and len(Y_test.unique())==2:
            benchmark_df = test_x[config_dict['Benchmark_cols'].keys()]
            benchmark_df.columns = benchmark_df.columns.to_series().map(config_dict['Benchmark_cols'])
            benchmark_df_names = benchmark_df.columns.tolist()
            min_max_scaler = MinMaxScaler()
            benchmark_df = min_max_scaler.fit_transform(benchmark_df)
            benchmark_df = pd.DataFrame(benchmark_df)
            benchmark_df.columns = benchmark_df_names
            #test_x.rename(columns=config_dict['Benchmark_cols'], inplace=True)
            class_weights = np.array([class_weights[i] for i in  Y_test])
            print(f"{so} class shape: {Y_test.value_counts()}")
            y_score = get_prediction(clf, test_x)
            get_matrix(y_score, Y_test, so, outdir)
            benchmark_df['DITTO'] = y_score
            roc_scores_so, prc_scores_so, f1_scores_so = get_roc_plot(so, benchmark_df, Y_test, outdir, class_weights)
            roc_scores[so].update(roc_scores_so)
            prc_scores[so].update(prc_scores_so)
            f1_scores[so].update(f1_scores_so)
            get_SHAP(test_x, clf, so, outdir, feature_names)

    pd.DataFrame(roc_scores).to_csv(f"{outdir}/NN_roc_scores.csv")
    pd.DataFrame(prc_scores).to_csv(f"{outdir}/NN_prc_scores.csv")
    pd.DataFrame(f1_scores).to_csv(f"{outdir}/NN_f1_scores.csv")
    return None

def data_parsing(test_x, config_dict):
    # Load data
    X_test = pd.read_csv(test_x)
    Y_test = X_test['class']
    X_test = X_test.drop(config_dict["train_cols"], axis=1)
    X_test = X_test.drop("class", axis=1)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)
    feature_names = X_test.columns.tolist()

    class_weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(Y_test),y=Y_test)
    class_weights = {i:w for i,w in enumerate(class_weights)}
    Y_test = label_binarize(
    Y_test.values, classes=list(np.unique(Y_test))
        ).ravel()
    X_test['class'] = Y_test
    print("Data Loaded!")

    return X_test, feature_names, class_weights

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
    X_test, feature_names, class_weights = data_parsing(
        ARGS.test_x, config_dict
    )

    run_test(X_test, out_dir, clf, config_dict, feature_names, class_weights)
