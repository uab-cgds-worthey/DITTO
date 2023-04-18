# python src/cohort/ranks.py --json /data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json -id /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/annotated_vcf -o cosine_ranks.csv

import json
import pandas as pd
import argparse
import os

def main(args):

    json_file = json.load(open(args.json, "r"))

    with open(f"{args.input_dir}/{args.output}", "w") as f:

        f.write(f"PROBANDID,[CHROM,POS,REF,ALT],SYMBOL,Ditto,cosine,projection,jaccard,combined_cosine,combined_projection,combined_jaccard,Rank['Ditto', 'cosine', 'jaccard', 'projection', 'combined_cosine', 'combined_jaccard', 'combined_projection']\n")

    #rank_list = []
    for samples in json_file["train"].keys():
        if "PROBAND" in samples:


            genes = pd.read_csv(
                f"{args.input_dir}/train/{samples}/Ditto_Hazel_fixed_no_bias.csv"
            )  # , sep=':')

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


                var_rank = []
                for method in ['Ditto', 'cosine', 'jaccard', 'projection', 'combined_cosine', 'combined_jaccard', 'combined_projection']:
                    if method == 'Ditto':
                        genes = genes.sort_values(by = 'Ditto', axis=0, ascending=False).reset_index(drop=True)
                    else:
                        genes = genes.sort_values(by = [method,'Ditto'], axis=0, ascending=[False,False]).reset_index(drop=True)
                    genes = genes.drop_duplicates(
                        subset=["CHROM", "POS", "ALT", "REF"], keep="first"
                        ).reset_index(drop=True)
                    rank = (
                        genes.loc[
                            (genes["CHROM"] == variants[0])
                            & (genes["POS"] == int(variants[1]))
                            & (genes["ALT"] == variants[3])
                            & (genes["REF"] == variants[2])
                        ].index
                    ) + 1
                    var_rank.append(rank.tolist()[0])

                #rank_list = [*rank_list, *rank]  # unpack both iterables in a list literal
                with open(f"{args.input_dir}/{args.output}", "a") as f:
                    f.write(
                         f"{samples}, {variants}, {genes.loc[rank-1]['SYMBOL'].values}, {genes.loc[rank-1]['Ditto'].values[0]}, {genes.loc[rank-1]['cosine'].values[0]}, {genes.loc[rank-1]['jaccard'].values[0]}, {genes.loc[rank-1]['projection'].values[0]}, {genes.loc[rank-1]['combined_cosine'].values[0]}, {genes.loc[rank-1]['combined_jaccard'].values[0]}, {genes.loc[rank-1]['combined_projection'].values[0]}, {var_rank}\n"
                    )
            del genes, rank, variants

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-id",
        type=str,
        required=True,
        help="Input raw annotated file with path.",
    )
    parser.add_argument(
        "--json", type=str, required=True, help="Input raw annotated file with path."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="ranks.csv", help="Output csv filename only"
    )
    args = parser.parse_args()

    main(args)
