import json
import pandas as pd
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    "-id",
    type=str,
    required=True,
    help="Input raw annotated file with path.")
parser.add_argument(
    "--json",
    type=str,
    required=True,
    help="Input raw annotated file with path.")
parser.add_argument(
    "--output",
    "-o",
    type=str,
    default="ranks.csv",
    help="Output csv filename only")
args = parser.parse_args()

#json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json", 'r'))
json_file = json.load(open(args.json, 'r'))
#
with open(f"{args.input_dir}/{args.output}", 'w') as f:
        f.write(f"PROBANDID,SYMBOL,Exomiser,Ditto,Combined,Rank\n")
rank_list = []
for samples in json_file['train'].keys():
        if "PROBAND" in samples:
            #print(samples)
            for i in range(len(json_file['train'][samples]["solves"])):
                gene = str(json_file['train'][samples]["solves"][i]["Gene"])
                #print(gene)
                #print('Reading Exomiser scores...')
                all_files = glob.glob(os.path.join(f"/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/exomiser/hpo_original/train/{samples}", "*.tsv"))
                exo_scores = pd.concat((pd.read_csv(f, sep='\t') for f in all_files), ignore_index=True)
                exo_scores = exo_scores[['#GENE_SYMBOL', 'ENTREZ_GENE_ID', 'EXOMISER_GENE_PHENO_SCORE']]
                exo_scores = exo_scores.sort_values('EXOMISER_GENE_PHENO_SCORE', ascending=False)
                exo_scores = exo_scores.drop_duplicates(subset=['#GENE_SYMBOL'], keep='first').reset_index(drop=True)
                rank = (exo_scores.loc[(exo_scores['#GENE_SYMBOL'] == gene)].index)+1
                #     #rank = ((genes.loc[(genes['CHROM'] == variants[0]) & (genes['POS'] == int(variants[1])) & (genes['ALT'] == variants[3]) & (genes['REF'] == variants[2])].index)+1)
                rank_list = [*rank_list, *rank]  # unpack both iterables in a list literal
                with open(f"{args.input_dir}/{args.output}", 'a') as f:
                    #f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['E'].values}, {genes.loc[rank-1]['D'].values}, {genes.loc[rank-1]['P'].values}, {rank.tolist()}\n")
                    f.write(f"{samples},{gene},{exo_scores.loc[rank-1]['EXOMISER_GENE_PHENO_SCORE'].values},{rank.tolist()}\n")

with open(f"{args.input_dir}/{args.output}", 'a') as f:
        f.write(f"\nList,{rank_list}\n")
        f.write(f"Rank-1,{sum(i < 2 for i in rank_list)}\n")
        f.write(f"Rank-10,{sum(i < 11 for i in rank_list)}\n")
        f.write(f"Rank-50,{sum(i < 51 for i in rank_list)}\n")
        f.write(f"Rank-100,{sum(i < 101 for i in rank_list)}\n")
        f.write(f"Rank-500,{sum(i < 501 for i in rank_list)}\n")
        f.write(f"Rank-1000,{sum(i < 1001 for i in rank_list)}\n")
        f.write(f"Rank-10000,{sum(i < 10001 for i in rank_list)}\n")
        f.write(f"#Predictions,{len(rank_list)}\n")
