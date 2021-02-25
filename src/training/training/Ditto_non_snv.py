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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import yaml
import os
from sklearn.ensemble import VotingClassifier
os.chdir('/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/')

print("Loading data....")

with open("../../configs/columns_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

df = pd.read_csv('merged_sig_norm_class_vep-annotated.tsv', sep='\t')
print('Data Loaded!')
df = df[config_dict['columns']]
df=df[(df['Alternate Allele'].str.len() > 1) | (df['Reference Allele'].str.len() > 1)]
print(f'Total non SNVs: {df.shape[0]}')
df = df.drop(config_dict['var'], axis=1)

X_train= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
X_train.dropna(axis=0, thresh=(X_train.shape[1]*0.3), inplace=True)  #thresh=(df.shape[1]*0.3),   how='all',
print('\nhgmd_class:\n', X_train['hgmd_class'].value_counts())
y = X_train.hgmd_class.str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
y = y.str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
y = y.str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
y = y.str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
Y_train = label_binarize(y.values, classes=['low_impact', 'high_impact'])
X_train = X_train.drop('hgmd_class', axis=1)
X_train = pd.get_dummies(X_train, prefix_sep='_')
df1=pd.DataFrame()
for key in tqdm(config_dict['non_snv_columns']):
    if key in X_train.columns:
        df1[key] = X_train[key].fillna(config_dict['non_snv_columns'][key]).astype('float64')
    else:
        df1[key] = config_dict['non_snv_columns'][key]

X_train = df1
del df1
feature_names = X_train.columns.tolist()
print(f'non SNVs to train: {X_train.shape[0]}')
X_train = X_train.values

#training data
X_test = df.loc[df['hgmd_class'].isin(config_dict['Clinsig_test'])]
#X_train= df.loc[df['hgmd_class'].isin(config_dict['Clinsig_train'])]
#df = df[(df['Alternate Allele'].str.len() > 1) | (df['Reference Allele'].str.len() > 1)]
print('\nhgmd_class:\n', X_test['hgmd_class'].value_counts())
y = X_test.hgmd_class.str.replace(r'DFP','high_impact').str.replace(r'DM\?','high_impact').str.replace(r'DM','high_impact')
y = y.str.replace(r'Pathogenic/Likely_pathogenic','high_impact').str.replace(r'Likely_pathogenic','high_impact').str.replace(r'Pathogenic','high_impact')
y = y.str.replace(r'DP','low_impact').str.replace(r'FP','low_impact')
y = y.str.replace(r'Benign/Likely_benign','low_impact').str.replace(r'Likely_benign','low_impact').str.replace(r'Benign','low_impact')
Y_test = label_binarize(y.values, classes=['low_impact', 'high_impact'])
X_test = X_test.drop('hgmd_class', axis=1)
X_test = pd.get_dummies(X_test, prefix_sep='_')
df1=pd.DataFrame()
for key in tqdm(config_dict['non_snv_columns']):
    if key in X_test.columns:
        df1[key] = X_test[key].fillna(config_dict['non_snv_columns'][key]).astype('float64')
    else:
        df1[key] = config_dict['non_snv_columns'][key]

X_test = df1
del df1,df
#feature_names = X_train.columns.tolist()
print(f'non SNVs to test: {X_test.shape[0]}')
X_test = X_test.values

model = VotingClassifier(estimators=[
        ('DecisionTreeClassifier',DecisionTreeClassifier(class_weight='balanced')),
        ('LogisticRegression',LogisticRegression(class_weight='balanced')),
        ('RandomForestClassifier',RandomForestClassifier(class_weight='balanced')),
        ('AdaBoostClassifier',AdaBoostClassifier()),
	    ('ExtraTreesClassifier',ExtraTreesClassifier(class_weight='balanced')),
        ('BalancedRandomForestClassifier',BalancedRandomForestClassifier()),
        ('GaussianNB',GaussianNB()),
        ('LinearDiscriminantAnalysis',LinearDiscriminantAnalysis())
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
background = shap.kmeans(X_train, 6)
explainer = shap.KernelExplainer(model.predict, background)
background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
shap_values = explainer.shap_values(background)
plt.figure()
shap.summary_plot(shap_values, background, feature_names, show=False)
#shap.plots.waterfall(shap_values[0], max_display=15)
plt.savefig("./models/Ditto_non_snv_features.pdf", format='pdf', dpi=1000, bbox_inches='tight')
pickle.dump(model, open("./models/Ditto_non_snv.pkl", 'wb'))