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

    image1 = Image.open("./data/processed/StackingClassifier_dbnsfp_features.jpg")


    return pred, clf, image1

def main():

    col1, col2, col3 = st.columns(3)

    pred, clf, image1 = load_data()
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
    col2.subheader(f"Ditto score = {ditto['Ditto'].values[0]}\n{row_idx}")

    with gzip.open("./data/processed/all_data_custom-dbnsfp.csv.gz", "rt") as vcffp:
        for cnt, line in enumerate(vcffp):
            if cnt == row_idx+1:
                #col2.write(np.array(line.split(',')))
                var_details = line.split(',')[:-106]
                col2.write(line.split(',')[-106:])
                col2.write(line.split(',')[:-106])
                break

    cols = ['Chromosome', 'position', 'ref_allele', 'alt_allele', 'strand', 'Gene', 'Transcript', 'Clinvar_significance', 'Clinvar_review', 'Clinvar_significance', 'Clinvar_significance']
    #splits = re.split(':|>',option)
    #col2.write(f"Chromosome: {splits[0]}")
    #col2.write(f"position: {splits[1]}")
    #col2.write(f"ref_allele: {splits[2]}")
    #col2.write(f"alt_allele: {splits[3]}")
    #col2.write(f"Gene: {splits[4]}")
    #col2.write(f"Transcript: {splits[5]}")
    #col2.write(f"Ditto deleterious: {var_list[option][1]}")

if __name__ == "__main__":
    main()
