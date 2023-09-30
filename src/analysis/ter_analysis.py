import pandas as pd
import obonet
import networkx as nx
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
from tqdm import tqdm
from pathlib import Path
import json

def get_gene_df():
    gene2hpo = {}
    alt_id = {}
    graph = obonet.read_obo(
        "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"
    )
    for id_, node in graph.nodes(data=True):
            for alt in node.get('alt_id', []):
                alt_id[alt] = id_

    associations_url = "http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt"
    associations = pd.read_csv(associations_url, sep='\t')
    associations = associations[['gene_symbol','hpo_id']]

    gene2hpo = dict()
    for ind in associations.index:
      gene_name = associations["gene_symbol"][ind]
      hp_term = associations["hpo_id"][ind]
      if gene_name not in gene2hpo:
          gene2hpo[gene_name] = list([])
      gene2hpo[gene_name].append(hp_term)
    #gene2hpo[gene_name] = gene2hpo[gene_name] + list(nx.neighbors(graph, hp_term))
    gene2hpo[gene_name] = gene2hpo[gene_name] + list(nx.predecessor(graph, hp_term))
    #gene2hpo[gene_name] = gene2hpo[gene_name] + list(graph.predecessors(hp_term)) + list(graph.successors(hp_term))
    gene2hpo[gene_name] = list(
        dict.fromkeys(gene2hpo[gene_name])
    )  # check for duplicate values and drop them
    gene2hpo[gene_name] = list(
        filter(None, gene2hpo[gene_name])
    )  # check for null values and drop them
    # gene2hpo[gene_name] = comparison.parse_terms(gene2hpo[gene_name])

    gene2hpo = {"Genes": list(gene2hpo.keys()), "HPOs": list(gene2hpo.values())}
    gene2hpo = pd.DataFrame(gene2hpo)
    return gene2hpo, graph, alt_id

def parse_terms(alt_terms, in_terms):
        """ This function parses through terms that are provided, then it returns a new list with updated HPO terms. """
        new_terms = []
        for term in in_terms:
            if term in list(alt_terms.keys()):
                new_terms.append(alt_terms[term])
            else:
                new_terms.append(term)
        return new_terms


def hpo_file():
    # Specify the path to your JSON file
    file_path = 'data/external/train_test_metadata_nonGeneticHPOsRemoved.json'

    # Open the JSON file for reading
    with open(file_path, 'r') as json_file:
        # Use json.load() to parse the JSON data from the file
        metadata = json.load(json_file)
    return metadata


def hazel(alt_id,gene_2_hpo,hpo_terms):
    input_phenotypes = list(hpo_terms)
    #input_phenotypes =['HP:0002020', 'HP:0008738', 'HP:0000218', 'HP:0009747', 'HP:0100814', 'HP:0001773', 'HP:0200055', 'HP:0003128', 'HP:0001252', 'HP:0012758', 'HP:0000707', 'HP:0001250', 'HP:0100547', 'HP:0007359', 'HP:0000326', 'HP:0000076']
    input_phenotypes = parse_terms(alt_id, input_phenotypes)
    input_phenotypes = list(dict.fromkeys(input_phenotypes))  # check for duplicate values and drop them
    term_len = len(input_phenotypes)
    ranked_genes = gene_2_hpo
    ranked_genes["score"] = [
                (len(list(set(input_phenotypes) & set(i)))) / term_len for i in ranked_genes["HPOs"].values
            ]
    ranked_genes = (
    ranked_genes[["Genes", "score"]]
                .sort_values(by="score", ascending=False)
                .reset_index(drop=True)
                )
    return ranked_genes


print("Generating Gene-Phene vectors from HPO...")
gene_2_hpo, graph, alt_id = get_gene_df()
metadata = hpo_file()
print("done!")
# Filter the dictionary to keep only dictionaries containing "pro" in keys or values
# probands = {key: subdict for key, subdict in {**metadata['test']}.items() if 'PROBAND' in key} # only test probands
probands = {key: subdict for key, subdict in {**metadata['train'], **metadata['test']}.items() if 'PROBAND' in key}
for proband in tqdm(probands):
    #print(f"Analysing {proband}...")
    ranked_genes = hazel(alt_id,gene_2_hpo,probands[proband]['hpo'].keys())
    p1 = pd.read_csv(f'data/external/CAGI_TR/{proband}_DITTO_scores.csv.gz',low_memory=False)
    p1 = p1.sort_values(by="DITTO", ascending=False).drop_duplicates(subset=['chrom', 'pos', 'ref_base', 'alt_base'], keep='first').reset_index(drop=True)
    p1['gnomad3.af'].fillna(0,inplace=True)
    merged = p1.merge(ranked_genes, left_on='gene', right_on = 'Genes', how='left')
    del p1, ranked_genes
    merged['combined'] = merged['DITTO'] + merged['score']
    merged = merged.sort_values(by="combined", ascending=False).reset_index(drop=True)
    ditto_sorted = merged.sort_values(by="DITTO", ascending=False).reset_index(drop=True)
    ditto_sorted = ditto_sorted[ditto_sorted['gene'].notna()]

    ditto_sorted[(ditto_sorted['gnomad3.af']<0.0005)].head(999).to_csv(f'data/processed/CAGI_analysis/{proband}_ditto100_vars.csv', index=False)
    del ditto_sorted

    #merged[(merged['gnomad3.af']<0.0005)].head(99).to_csv(f'data/processed/CAGI_analysis/{proband}_top100_vars.csv', index=False)
#
    #merged[(merged['combined']>0.9) & (merged['DITTO']>0.9) & (merged['gnomad3.af']<0.0005)].to_csv(f'data/processed/CAGI_analysis/{proband}_rare_filter.csv', index=False)
#
    #merged.loc[(merged['clinvar.sig'].isin(['Likely pathogenic', 'Pathogenic', 'Pathogenic/Likely pathogenic', 'Pathogenic/Likely pathogenic|other', 'Pathogenic|risk factor']))].to_csv(f'data/processed/CAGI_analysis/{proband}_interesting_vars.csv', index=False)
    #merged.loc[(merged['clinvar.sig'].isin(['Likely pathogenic', 'Pathogenic', 'Pathogenic/Likely pathogenic','risk factor', 'drug response','Likely risk allele', 'association|drug response|risk factor', 'association|drug response', 'Pathogenic/Likely pathogenic|other','protective|risk factor', 'Pathogenic|risk factor']))][['chrom', 'pos', 'ref_base', 'alt_base','protein_hgvs','consequence','clinvar.sig','clingen.disease','clingen.classification','Genes', 'score','gnomad3.af', 'DITTO', 'combined']].to_csv(f'data/processed/CAGI_analysis/{proband}_interesting_vars.csv', index=False)
