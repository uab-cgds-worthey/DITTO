#from numpy import mean
import numpy as np
import pandas as pd
import time
import warnings
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
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import gc
import os
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/train_test/')

TUNE_STATE_REFRESH_PERIOD = 10  # Refresh resources every 10 s

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
    return X_test, Y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var-tag",
        "-v",
        type=str,
        required=True,
        default='nssnv',
        help="The tag used when generating train/test data. Default:'nssnv'")

    args = parser.parse_args()
    var = args.var_tag

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

    if not os.path.exists('../tuning/'+var):
        os.makedirs('../tuning/'+var)
        
    print('Working with '+var+' dataset...')
    X_test,Y_test= data_parsing(var,config_dict)

    # prepare plots
    fig, [ax_roc, ax_prc] = plt.subplots(1, 2, figsize=(20, 10))

    for name in classifiers:
        with open(f"../tuning/{var}/{name}_{var}.joblib", 'rb') as f:
            clf = load(f)
        
        plot_precision_recall_curve(clf, X_test, Y_test, ax=ax_prc, name=name)
        plot_roc_curve(clf, X_test, Y_test, ax=ax_roc, name=name)
    
    ax_roc.set_title('Receiver Operating Characteristic (ROC) curves')
    ax_prc.set_title('Precision Recall (PRC) curves')

    ax_roc.grid(linestyle='--')
    ax_prc.grid(linestyle='--')

    plt.legend()
    plt.title('Model performances on Testing data with filters ()')
    plt.savefig(f"../tuning/{var}/tuned_roc_{var}.pdf", format='pdf', dpi=1000, bbox_inches='tight')
    gc.collect()

