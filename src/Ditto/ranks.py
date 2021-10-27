import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    "-id",
    type=str,
    required=True,
<<<<<<< HEAD
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

with open(f"{args.input_dir}/{args.output}", 'w') as f:
        f.write(f"PROBANDID,[CHROM,POS,REF,ALT],SYMBOL,Exomiser,Ditto,Combined,Rank\n")
rank_list = []
for samples in json_file['train'].keys():
        if "PROBAND" in samples:
            genes = pd.read_csv(f"{args.input_dir}/train/{samples}/ditto_predictions.csv")#, sep=':')
            #genes = pd.read_csv(f"{args.input_dir}/train/{samples}/combined_predictions.csv")#, sep=':')
            #genes = genes.sort_values(by = ['E','P'], axis=0, ascending=[False,False], kind='quicksort', ignore_index=True)
            genes = genes.drop_duplicates(subset=['Chromosome','Position','Alternate Allele','Reference Allele'], keep='first').reset_index(drop=True)
            #genes = genes.drop_duplicates(subset=['CHROM','POS','ALT','REF'], keep='first').reset_index(drop=True)
            for i in range(len(json_file['train'][samples]["solves"])):
                variants = str('chr' + str(json_file['train'][samples]["solves"][i]["Chrom"]).split('.')[0] + ',' + str(json_file['train'][samples]["solves"][i]["Pos"]) + ',' + json_file['train'][samples]["solves"][i]["Ref"] + ',' + json_file['train'][samples]["solves"][i]["Alt"]).split(',')
                rank = ((genes.loc[(genes['Chromosome'] == variants[0]) & (genes['Position'] == int(variants[1])) & (genes['Alternate Allele'] == variants[3]) & (genes['Reference Allele'] == variants[2])].index)+1)
                #rank = ((genes.loc[(genes['CHROM'] == variants[0]) & (genes['POS'] == int(variants[1])) & (genes['ALT'] == variants[3]) & (genes['REF'] == variants[2])].index)+1)
                rank_list = [*rank_list, *rank]  # unpack both iterables in a list literal
                with open(f"{args.input_dir}/{args.output}", 'a') as f:
                    #f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['E'].values}, {genes.loc[rank-1]['D'].values}, {genes.loc[rank-1]['P'].values}, {rank.tolist()}\n")
                    f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['Ditto_Deleterious'].values}, {rank.tolist()}\n")

with open(f"{args.input_dir}/{args.output}", 'a') as f:
        f.write(f"\nList,{rank_list}\n")
        f.write(f"Rank-1,{sum(i < 2 for i in rank_list)}\n")
        f.write(f"Rank-5,{sum(i < 6 for i in rank_list)}\n")
        f.write(f"Rank-10,{sum(i < 11 for i in rank_list)}\n")
        f.write(f"Rank-20,{sum(i < 21 for i in rank_list)}\n")
        f.write(f"Rank-50,{sum(i < 51 for i in rank_list)}\n")
        f.write(f"Rank-100,{sum(i < 101 for i in rank_list)}\n")
        f.write(f"#Predictions,{len(rank_list)}\n")
=======
    help="Input raw annotated file with path.",
)
parser.add_argument(
    "--json", type=str, required=True, help="Input raw annotated file with path."
)
parser.add_argument(
    "--output", "-o", type=str, default="ranks.csv", help="Output csv filename only"
)
args = parser.parse_args()

# json_file = json.load(open("/data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json", 'r'))
json_file = json.load(open(args.json, "r"))

with open(f"{args.input_dir}/{args.output}", "w") as f:
    f.write(f"PROBANDID,[CHROM,POS,REF,ALT],SYMBOL,Exomiser,Ditto,Combined,Rank\n")
rank_list = []
for samples in json_file["train"].keys():
    if "PROBAND" in samples:
        # genes = pd.read_csv(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/filter_vcf_by_DP6_AB/train/{samples}/combined_predictions.csv")#, sep=':')
        genes = pd.read_csv(
            f"{args.input_dir}/train/{samples}/combined_predictions.csv"
        )  # , sep=':')
        # genes = genes.sort_values(by = ['E','P'], axis=0, ascending=[False,False], kind='quicksort', ignore_index=True)
        # genes = genes.drop_duplicates(subset=['Chromosome','Position','Alternate Allele','Reference Allele'], keep='first').reset_index(drop=True)
        genes = genes.drop_duplicates(
            subset=["CHROM", "POS", "ALT", "REF"], keep="first"
        ).reset_index(drop=True)
        for i in range(len(json_file["train"][samples]["solves"])):
            variants = str(
                "chr"
                + str(json_file["train"][samples]["solves"][i]["Chrom"]).split(".")[0]
                + ","
                + str(json_file["train"][samples]["solves"][i]["Pos"])
                + ","
                + json_file["train"][samples]["solves"][i]["Ref"]
                + ","
                + json_file["train"][samples]["solves"][i]["Alt"]
            ).split(",")
            # rank = ((genes.loc[(genes['Chromosome'] == variants[0]) & (genes['Position'] == int(variants[1])) & (genes['Alternate Allele'] == variants[3]) & (genes['Reference Allele'] == variants[2])].index)+1)
            rank = (
                genes.loc[
                    (genes["CHROM"] == variants[0])
                    & (genes["POS"] == int(variants[1]))
                    & (genes["ALT"] == variants[3])
                    & (genes["REF"] == variants[2])
                ].index
            ) + 1
            rank_list = [*rank_list, *rank]  # unpack both iterables in a list literal
            with open(f"{args.input_dir}/{args.output}", "a") as f:
                f.write(
                    f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['E'].values}, {genes.loc[rank-1]['D'].values}, {genes.loc[rank-1]['P'].values}, {rank.tolist()}\n"
                )
                # f.write(f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['Ditto_Deleterious'].values}, {rank.tolist()}\n")

with open(f"{args.input_dir}/{args.output}", "a") as f:
    # f.write(f"\nList,{rank_list}\n")
    f.write(f"Rank-1,{sum(i < 2 for i in rank_list)}\n")
    f.write(f"Rank-5,{sum(i < 6 for i in rank_list)}\n")
    f.write(f"Rank-10,{sum(i < 11 for i in rank_list)}\n")
    f.write(f"Rank-20,{sum(i < 21 for i in rank_list)}\n")
    f.write(f"Rank-50,{sum(i < 51 for i in rank_list)}\n")
    f.write(f"Rank-100,{sum(i < 101 for i in rank_list)}\n")
    f.write(f"#Predictions,{len(rank_list)}\n")
>>>>>>> bcf8822dc39ff4415e7d8b84138b7463ca901555
