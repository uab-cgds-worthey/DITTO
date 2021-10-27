<<<<<<< HEAD
#from numpy import mean
=======
# from numpy import mean
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
import numpy as np
import pandas as pd
import time
import warnings
import argparse
<<<<<<< HEAD
warnings.simplefilter("ignore")
from joblib import dump, load
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
=======

warnings.simplefilter("ignore")
from joblib import dump, load

# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFE
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

# from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import gc
import os


TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

<<<<<<< HEAD
def data_parsing(var,config_dict):
    #Load data
    X_test = pd.read_csv(f'test_{var}/merged_data-test_{var}.csv')
    #var = X_test[config_dict['ML_VAR']]
    X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
    #feature_names = X_test.columns.tolist()
    X_test = X_test.values
    Y_test = pd.read_csv(f'test_{var}/merged_data-y-test_{var}.csv')
    print('Data Loaded!')
    #Y = pd.get_dummies(y)
    Y_test = label_binarize(Y_test.values, classes=['low_impact', 'high_impact']).ravel()  
    
    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    # explain all the predictions in the test set
    #background = shap.kmeans(X_train, 10)
=======

def data_parsing(var, config_dict):
    # Load data
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
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
    return X_test, Y_test


if __name__ == "__main__":
<<<<<<< HEAD
    os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test/')
=======
    os.chdir(
        "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test/"
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        required=True,
<<<<<<< HEAD
        default='nssnv',
        help="The tag used when generating train/test data. Default:'nssnv'")
=======
        default="nssnv",
        help="The tag used when generating train/test data. Default:'nssnv'",
    )
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

    args = parser.parse_args()
    var = args.var_tag

<<<<<<< HEAD
    #Classifiers I wish to use
    classifiers = [
        	"DecisionTree",
            "RandomForest",
            "BalancedRF",
            "AdaBoost",
            "ExtraTrees",
            "GaussianNB",
            "LDA",
            "GradientBoost",
            "MLP",
            "StackingClassifier"
    ]
    
    
    with open("../../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    if not os.path.exists('../models/'+var):
        os.makedirs('../models/'+var)
        
    print('Working with '+var+' dataset...')
    X_test,Y_test= data_parsing(var,config_dict)
=======
    # Classifiers I wish to use
    classifiers = [
        "DecisionTree",
        "RandomForest",
        "BalancedRF",
        "AdaBoost",
        "ExtraTrees",
        "GaussianNB",
        "LDA",
        "GradientBoost",
        "MLP",
        "StackingClassifier",
    ]

    with open("../../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    if not os.path.exists("../models/" + var):
        os.makedirs("../models/" + var)

    print("Working with " + var + " dataset...")
    X_test, Y_test = data_parsing(var, config_dict)
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555

    # prepare plots
    fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f"Model performances on Testing data", fontsize=20)

    for name in classifiers:
<<<<<<< HEAD
        
        with open(f"../models/{var}/{name}_{var}.joblib", 'rb') as f:
            clf = load(f)
        
        plot_precision_recall_curve(clf, X_test, Y_test, ax=ax_prc, name=name)
        plot_roc_curve(clf, X_test, Y_test, ax=ax_roc, name=name)
    
    ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
    ax_prc.set_title('Precision Recall (PRC) curves')

    ax_roc.grid(linestyle='--')
    ax_prc.grid(linestyle='--')

    plt.legend()
    plt.savefig(f"../models/{var}/roc_{var}.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    gc.collect()


=======

        with open(f"../models/{var}/{name}_{var}.joblib", "rb") as f:
            clf = load(f)

        plot_precision_recall_curve(clf, X_test, Y_test, ax=ax_prc, name=name)
        plot_roc_curve(clf, X_test, Y_test, ax=ax_roc, name=name)

    ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
    ax_prc.set_title("Precision Recall (PRC) curves")

    ax_roc.grid(linestyle="--")
    ax_prc.grid(linestyle="--")

    plt.legend()
    plt.savefig(
        f"../models/{var}/roc_{var}.pdf", format="pdf", dpi=1000, bbox_inches="tight"
    )
    gc.collect()
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
