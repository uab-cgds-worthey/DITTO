import json
import pandas as pd

json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata.json", 'r'))

for samples in json_file['train'].keys():
        if "PROBAND" in samples and "TRAIN" in samples:
            genes = pd.read_csv(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/predictions/{'train'}/{samples}/combined_predictions_0_5.csv")#, sep=':')
            #genes = genes.sort_values(by = ['E','P'], axis=0, ascending=[False,False], kind='quicksort', ignore_index=True)
            #genes = genes.drop_duplicates(subset=['Chromosome','Position','Alternate Allele','Reference Allele'], keep='first').reset_index(drop=True)
            genes = genes.drop_duplicates(subset=['CHROM','POS','ALT','REF'], keep='first').reset_index(drop=True)
            variants = str('chr' + str(json_file['train'][samples]["solves"][0]["Chrom"]).split('.')[0] + ',' + str(json_file['train'][samples]["solves"][0]["Pos"]) + ',' + json_file['train'][samples]["solves"][0]["Ref"] + ',' + json_file['train'][samples]["solves"][0]["Alt"]).split(',')
            #rank = ((genes.loc[(genes['Chromosome'] == variants[0]) & (genes['Position'] == int(variants[1])) & (genes['Alternate Allele'] == variants[3]) & (genes['Reference Allele'] == variants[2])].index)+1).tolist()
            rank = ((genes.loc[(genes['CHROM'] == variants[0]) & (genes['POS'] == int(variants[1])) & (genes['ALT'] == variants[3]) & (genes['REF'] == variants[2])].index)+1).tolist()
            with open("Combined_ranking_0_5.csv", 'a') as f:
                    f.write(f"{samples}, {variants}, {rank}\n")

            #print(f'{i}:{samples}: {variant}', file=open("Ditto_ranking.csv", "a"))
