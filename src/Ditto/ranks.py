import json
import pandas as pd

json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata.json", 'r'))

for samples in json_file['train'].keys():
        if "PROBAND" in samples and "TRAIN" in samples:
            genes = pd.read_csv(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/predictions/{'train'}/{samples}/predictions_500.csv")
            variants = str('chr' + str(json_file['train'][samples]["solves"][0]["Chrom"]).split('.')[0] + ',' + str(json_file['train'][samples]["solves"][0]["Pos"]) + ',' + json_file['train'][samples]["solves"][0]["Ref"] + ',' + json_file['train'][samples]["solves"][0]["Alt"]).split(',')
            rank = ((genes.loc[(genes['Chromosome_x'] == variants[0]) & (genes['Position_x'] == int(variants[1])) & (genes['Alternate Allele_x'] == variants[3]) & (genes['Reference Allele_x'] == variants[2])].index)+1).tolist()
            with open("Ditto_ranking.csv", 'a') as f:
                    f.write(f"{samples}, {variants}, {rank}\n")
    
            #print(f'{i}:{samples}: {variant}', file=open("Ditto_ranking.csv", "a"))
