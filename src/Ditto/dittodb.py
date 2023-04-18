#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python slurm-launch.py --exp-name Training --command "python Ditto/dittodb.py -i /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/all_data_custom-dbnsfp.csv -O /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto" --partition largemem --mem 150G

import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load, dump
import argparse
import shap
import numpy as np
import matplotlib.pyplot as plt
import functools
print = functools.partial(print, flush=True)
from sklearn.preprocessing import label_binarize
from sklearn import metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input csv file with path for predictions",
    )
    parser.add_argument(
        "--out_dir",
        "-O",
        type=str,
        default=".",
        help="Output directory path",
    )

    args = parser.parse_args()

    print("Loading data and Ditto model....")

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/configs/col_config.yaml"
    ) as fh:
        config_dict = yaml.safe_load(fh)

    ditto_db = {}

    X = pd.read_csv(args.input)
    ditto_db['dbnsfp'] = X
    X_test = X
    var = X_test[config_dict["var"]]
    X_test = X_test.drop(config_dict["var"], axis=1)
    X_test = X_test.values

    with open(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/models_custom/dbnsfp/StackingClassifier_dbnsfp.joblib",
        "rb",
    ) as f:
        clf = load(f)

    X_train = pd.read_csv(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_custom_data-dbnsfp.csv")
    # var = X_train[config_dict['ML_VAR']]
    X_train = X_train.drop(config_dict["var"], axis=1)
    feature_names = X_train.columns.tolist()
    ditto_db['feature_names'] = feature_names
    #X_train = X_train.sample(frac=1).reset_index(drop=True)
    X_train = X_train.values
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(clf.predict, background)
    ditto_db['explainer'] = explainer

    del X_train
    shap_values = explainer.shap_values(X_test)

    ditto_db['shap_values'] = shap_values

    plt.figure()
    shap.summary_plot(shap_values, background, feature_names, max_display = 50, show=False)
    del background, shap_values, feature_names
    # shap.plots.waterfall(shap_values[0], max_display=15)
    plt.savefig(
        f"{args.out_dir}/shap_features.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )

    print('Ditto Loaded!\nRunning predictions.....')

    y_score = clf.predict_proba(X_test)
    del X_test, clf
    pred = pd.DataFrame(y_score, columns=["Ditto_Benign", "Ditto_Deleterious"])

    overall = pd.concat([var, pred], axis=1)
    ditto_db['ditto_predictions'] = overall
    overall.to_csv(args.out_dir + "ditto_predictions.csv.gz", index=False,
           compression="gzip")
    del y_score, overall

    print('writing to database...')
    with open(
        f"{args.out_dir}/dittoDB.joblib",
        "wb",
    ) as f:
        dump(ditto_db, f, compress="lz4")
    print('Database storage complete!\nBenchmarking Ditto....')

    #Benchmarking
    X = pd.concat([X, pred['Ditto_Deleterious']], axis=1)
    del pred, ditto_db

    benchmark_columns = ['Ditto_Deleterious','SIFT_score','MutationAssessor_score','CADD_phred','DANN_score','DEOGEN2_score','LRT_score','M-CAP_score','MetaLR_score','MetaSVM_score','MetaRNN_score','ClinPred_score','MutPred_score','VEST4_score','PrimateAI_score','clinvar_clnsig']
    X = X[benchmark_columns]
    X.columns = ['Ditto','SIFT','MutationAssessor','CADD','DANN','DEOGEN2','LRT','M-CAP','MetaLR','MetaSVM','MetaRNN','ClinPred','MutPred','VEST4','PrimateAI','clinvar']
    X = X.loc[X['clinvar'].isin(config_dict['BenchmarkSignificance'])]
    X = X.dropna(axis=0, subset=['clinvar']).reset_index(drop=True)
    Y_test = X['clinvar'].replace(r'Pathogenic/Likely_pathogenic,_other','1').replace(r'Pathogenic/Likely_pathogenic','1').str.replace(r'Likely_pathogenic','1').replace(r'Pathogenic,_other','1').replace(r'Pathogenic,_drug_response','1').replace(r'Pathogenic','1').replace(r'Benign/Likely_benign','0').replace(r'Likely_benign','0').replace(r'Benign','0').astype('int8')

    X = X.drop('clinvar', axis=1)

    plt.figure().clf()
    #plt.suptitle("Benchmarking damage prediction tools", fontsize=10)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) curves")
    plt.grid(linestyle="--")

    for name in list(X.columns):
        fpr, tpr, thresh = metrics.roc_curve(Y_test, X[name].fillna(0).values)
        auc = metrics.roc_auc_score(Y_test, X[name].fillna(0).values)
        plt.plot(fpr,tpr,label=str(name)+", auc="+str(auc))


    plt.legend()
    plt.savefig(
        f"{args.out_dir}/ROC.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )

    plt.figure().clf()
    #plt.suptitle("Benchmarking damage prediction tools", fontsize=10)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall (PRC) curves")
    plt.grid(linestyle="--")

    for name in list(X.columns):
        precision, recall, thresholds = metrics.precision_recall_curve(Y_test, X[name].fillna(0).values)
        prc = metrics.average_precision_score(Y_test, X[name].fillna(0).values)
        plt.plot(recall,precision,label=str(name)+", prc="+str(prc))


    plt.legend()
    plt.savefig(
        f"{args.out_dir}/PRC.pdf",
        format="pdf",
        dpi=1000,
        bbox_inches="tight",
    )


