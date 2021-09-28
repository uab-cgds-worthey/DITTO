import json
import pandas as pd

json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata.json", 'r'))

with open("ranks1.csv", 'w') as f:
        f.write(f"PROBANDID,[CHROM,POS,REF,ALT],SYMBOL,Exomiser,Ditto,Combined,Rank\n")

for samples in json_file['train'].keys():
        if "PROBAND" in samples and "TRAIN" in samples:
            genes = pd.read_csv(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/predictions/{'train'}/{samples}/ditto_predictions.csv")#, sep=':')
            #genes = genes.sort_values(by = ['E','P'], axis=0, ascending=[False,False], kind='quicksort', ignore_index=True)
            genes = genes.drop_duplicates(subset=['Chromosome','Position','Alternate Allele','Reference Allele'], keep='first').reset_index(drop=True)
            #genes = genes.drop_duplicates(subset=['CHROM','POS','ALT','REF'], keep='first').reset_index(drop=True)
            for i in range(len(json_file['train'][samples]["solves"])):
                variants = str('chr' + str(json_file['train'][samples]["solves"][i]["Chrom"]).split('.')[0] + ',' + str(json_file['train'][samples]["solves"][i]["Pos"]) + ',' + json_file['train'][samples]["solves"][i]["Ref"] + ',' + json_file['train'][samples]["solves"][i]["Alt"]).split(',')
                rank = ((genes.loc[(genes['Chromosome'] == variants[0]) & (genes['Position'] == int(variants[1])) & (genes['Alternate Allele'] == variants[3]) & (genes['Reference Allele'] == variants[2])].index)+1)
                #rank = ((genes.loc[(genes['CHROM'] == variants[0]) & (genes['POS'] == int(variants[1])) & (genes['ALT'] == variants[3]) & (genes['REF'] == variants[2])].index)+1)
                with open("ditto_ranks.csv", 'a') as f:
                    #f.write(f"{genes.loc[rank].values}\n")
                    #f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['E'].values}, {genes.loc[rank-1]['D'].values}, {genes.loc[rank-1]['P'].values}, {rank.tolist()}\n")
                    f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['Ditto_Deleterious'].values}, {rank.tolist()}\n")

            #print(f'{i}:{samples}: {variant}', file=open("Ditto_ranking.csv", "a"))
