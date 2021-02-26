import numpy as np
import pandas as pd
import pickle
import shap
from tqdm import tqdm
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, roc_auc_score, accuracy_score, confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.ensemble import VotingClassifier
import warnings
warnings.simplefilter("ignore")
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

print("Loading data....")

with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

X_train = pd.read_csv('train_non_snv/merged_data-train_non_snv.csv')
#var = X_train[config_dict['ML_VAR']]
X_train = X_train.drop(config_dict['ML_VAR'], axis=1)
feature_names = X_train.columns.tolist()
X_train = X_train.values
Y_train = pd.read_csv('train_non_snv/merged_data-y-train_non_snv.csv')
Y_train = label_binarize(Y_train.values, classes=['low_impact', 'high_impact']) 


X_test = pd.read_csv('test_non_snv/merged_data-test_non_snv.csv')
#var = X_test[config_dict['ML_VAR']]
X_test = X_test.drop(config_dict['ML_VAR'], axis=1)
#feature_names = X_test.columns.tolist()
X_test = X_test.values
Y_test = pd.read_csv('test_non_snv/merged_data-y-test_non_snv.csv')
#Y = pd.get_dummies(y)
Y_test = label_binarize(Y_test.values, classes=['low_impact', 'high_impact'])
print('Data Loaded!')

#scaler = StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)

model = VotingClassifier(estimators=[
        ('DecisionTreeClassifier',DecisionTreeClassifier(class_weight='balanced')),
        ('SGDClassifier',SGDClassifier(class_weight='balanced', n_jobs=-1)),
        ('RandomForestClassifier',RandomForestClassifier(class_weight='balanced', n_jobs=-1)),
        ('AdaBoostClassifier',AdaBoostClassifier()),
	    ('ExtraTreesClassifier',ExtraTreesClassifier(class_weight='balanced', n_jobs=-1)),
        ('BalancedRandomForestClassifier',BalancedRandomForestClassifier()),
        ('GaussianNB',GaussianNB()),
        ('LinearDiscriminantAnalysis',LinearDiscriminantAnalysis()),
        ('MLPClassifier',MLPClassifier())
        #,('EasyEnsembleClassifier',EasyEnsembleClassifier()) 
        ], voting='hard', n_jobs=-1, verbose=1)

print(f'Training voting classifier...')
score = cross_validate(model, X_train, Y_train, cv=10, return_train_score=True, return_estimator=True, n_jobs=-1, verbose=1)
train_score, train_min, train_max = np.mean(score['train_score']), np.min(score['train_score']), np.max(score['train_score'])
test_score, test_min, test_max = np.mean(score['test_score']), np.mean(score['test_score']), np.mean(score['test_score'])
print(f'Picking best model...')
model = score['estimator'][np.argmax(score['test_score'])]
#model.fit(X_train, Y_train)
#score = model.score(X_train, Y_train)
print(f'Testing voting classifier...')
y_score = model.predict(X_test)
prc = precision_score(Y_test,y_score, average="weighted")
recall  = recall_score(Y_test,y_score, average="weighted")
roc_auc = roc_auc_score(Y_test, y_score)
accuracy = accuracy_score(Y_test, y_score)
matrix = confusion_matrix(Y_test, y_score,)
#print(f'Model: Ditto\nTrain_score(train_data): {score}')
print(f'Model: Ditto_non_snv\nTrain_mean_score(train_data)[min, max]: {train_score}[{train_min}, {train_max}]\nTest_mean_score(train_data)[min, max]: {test_score}[{test_min}, {test_max}]')
print(f'Precision(test_data): {prc}\nRecall: {recall}\nroc_auc: {roc_auc}\nAccuracy: {accuracy}\nConfusion_matrix[low_impact, high_impact]:\n{matrix}')

print(f'Calculating SHAP values...')
# explain all the predictions in the test set
background = shap.kmeans(X_train, 10)
explainer = shap.KernelExplainer(model.predict, background)
background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
shap_values = explainer.shap_values(background)
plt.figure()
shap.summary_plot(shap_values, background, feature_names, show=False)
#shap.plots.waterfall(shap_values[0], max_display=15)
plt.savefig("./models/Ditto_non_snv_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
pickle.dump(model, open("./models/Ditto_non_snv.pkl", 'wb'))