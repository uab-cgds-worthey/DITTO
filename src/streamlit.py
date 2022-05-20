import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import yaml
import warnings
warnings.simplefilter("ignore")
from joblib import load
import shap
import numpy as np
import matplotlib.pyplot as plt
import gzip
from PIL import Image
#import re

@st.cache(allow_output_mutation=True)
def load_data():
    #with gzip.open('./data/processed/ditto_predictions.csv.gz', 'rt') as fp:

    pred = pd.read_csv('./data/processed/ditto_predictions.csv.gz', usecols=['#chr', 'pos(1-based)', 'ref', 'alt', 'genename', 'Ensembl_transcriptid', 'Ditto_Deleterious'])
    pred.columns = ['Chromosome', 'position', 'ref_allele', 'alt_allele', 'Gene', 'Transcript', 'Ditto']
    pred.position = pred.position.astype('int64')

    with open(
        "./data/processed/StackingClassifier_dbnsfp.joblib",
        "rb",
    ) as f:
        clf = load(f)

    X_train = pd.read_csv('./data/processed/train_custom_data-dbnsfp.csv.gz')
    X_train = X_train.drop(['#chr','pos(1-based)','ref','alt','cds_strand','genename','Ensembl_transcriptid','clinvar_clnsig','Ensembl_geneid','Ensembl_proteinid','Uniprot_acc','clinvar_review','Interpro_domain'], axis=1)
    X_train = X_train.values
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(clf.predict_proba, background)
    del clf,X_train,background

    image1 = Image.open("./data/processed/StackingClassifier_dbnsfp_features.jpg")


    return pred, explainer, image1

def main():

    col1, col2, col3 = st.columns(3)

    pred, explainer, image1 = load_data()
    col1.subheader("Variant details:")
    col3.subheader("DITTO overall feature importances")
    col3.image(image1, caption="DITTO overall feature importances generated using SHAP")

    method = col1.radio(
     "Search method:",
     ('Variant', 'Gene' ))

    if method == 'Variant':
        chr = col1.selectbox('Please select a chromosome:',options = pred.Chromosome.unique(), help = '1-22,X,Y,M')
        pos = col1.selectbox('Please select a position:',options = pred[pred.Chromosome==chr]['position'].unique())
        ref = col1.selectbox('Please select a Reference allele:',options = pred[(pred.Chromosome==chr) & (pred.position==pos)]['ref_allele'].unique())
        alt = col1.selectbox('Please select an Alternate allele:',options = pred[(pred.Chromosome==chr) & (pred.position==pos) & (pred.ref_allele==ref)]['alt_allele'].unique())
        trans = col1.selectbox('Please select a Transcript:',options = pred[(pred.Chromosome==chr) & (pred.position==pos) & (pred.ref_allele==ref)  & (pred.alt_allele==alt)]['Transcript'].unique())
        ditto = pred[(pred.Chromosome==chr) & (pred.position==pos) & (pred.ref_allele==ref) & (pred.alt_allele==alt) & (pred.Transcript==trans)]
    else:
        gene = col1.selectbox('Please select a Gene:',options = pred.Gene.unique())
        trans = col1.selectbox('Please select a Transcript:',options = pred[pred.Gene==gene]['Transcript'].unique())
        pos = col1.selectbox('Please select a Position:',options = pred[(pred.Gene==gene) & (pred.Transcript==trans)]['position'].unique())
        ref = col1.selectbox('Please select a Reference allele:',options = pred[(pred.Gene==gene) & (pred.Transcript==trans) & (pred.position==pos)]['ref_allele'].unique())
        alt = col1.selectbox('Please select an Alternate allele:',options = pred[(pred.Gene==gene) & (pred.Transcript==trans) & (pred.position==pos) & (pred.ref_allele==ref)]['alt_allele'].unique())
        ditto = pred[(pred.Gene==gene) & (pred.Transcript==trans) & (pred.position==pos) & (pred.ref_allele==ref) & (pred.alt_allele==alt)]

    row_idx = ditto.index.values[0]
    col2.subheader(f"Ditto score = {ditto['Ditto'].values[0]}")#\n{row_idx}")

    annots = pd.read_csv('./data/processed/all_data_custom-dbnsfp.csv.gz',skiprows=range(1,row_idx), nrows=1)
    col2.write(f"Chromosome = {annots['#chr'].values[0]}")
    col2.write(f"Pos(1-based) = {annots['pos(1-based)'].values[0]}")
    col2.write(f"Reference allele = {annots['ref'].values[0]}")
    col2.write(f"Alternate allele = {annots['alt'].values[0]}")
    col2.write(f"Strand = {annots['cds_strand'].values[0]}")
    col2.write(f"Gene name = {annots['genename'].values[0]}")
    col2.write(f"Transcript = {annots['Ensembl_transcriptid'].values[0]}")
    col2.write(f"Clinvar significance = {annots['clinvar_clnsig'].values[0]}")
    annots = annots.drop(['#chr','pos(1-based)','ref','alt','cds_strand','genename','Ensembl_transcriptid','clinvar_clnsig','Ensembl_geneid','Ensembl_proteinid','Uniprot_acc','clinvar_review','Interpro_domain'], axis=1)
    #col2.write(annots)
    shap_values1 = explainer.shap_values(annots.iloc[0,:])
    plt.figure()
    shap.force_plot(explainer.expected_value[1], shap_values1[1], annots.iloc[0,:], matplotlib = True, show = True)
    col2.pyplot(plt)


if __name__ == "__main__":
    main()
