# python src/cohort/combine_scores.py --json /data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json --ditto /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/annotated_vcf --hazel /data/project/worthey_lab/projects/experimental_pipelines/tarun/uab-meter/data/processed/CAGI6

import pandas as pd
import warnings
import json
warnings.simplefilter("ignore")
import argparse
import os
import sys

def main(args):
    print("Loading cohort metadata file....")
    json_file = json.load(open(args.json, "r"))

    print("Loading Biomart file....")
    id_map = pd.read_csv(
                "/data/project/worthey_lab/temp_datasets_central/tarun/HGNC/biomart_9_23_21.txt",
                sep="\t",
            )

    for samples in json_file["train"].keys():
        if "2_PROBAND" in samples:
            print(f"Running sample:{samples}....")

            print("Loading Ditto file....")
            ditto = pd.read_csv(f"{args.ditto}/train/{samples}/ditto_predictions.csv")

            print("Loading Raw file....")
            raw = pd.read_csv(
                f"{args.ditto}/train/{samples}_vep-annotated_filtered.tsv",
                sep="\t",
                usecols=[
                    "SYMBOL",
                    "Chromosome",
                    "Position",
                    "Reference Allele",
                    "Alternate Allele",
                    "Gene",
                    "Feature",
                    "HGNC_ID",
                ],
            )
            print("Raw file loaded!")

            overall = pd.merge(
                raw,
                ditto,
                how="left",
                on=[
                    "Chromosome",
                    "Position",
                    "Alternate Allele",
                    "Reference Allele",
                    "Feature",
                ],
            )
            # print(overall.columns.values.tolist())
            del raw, ditto

            print("Reading Hazel scores...")
            hazel = pd.read_csv(f"{args.hazel}/{samples}/Hazel_{samples}.csv")
            #, usecols=["Genes","cosine","projection","jaccard"]
            id_map = id_map.merge(
                hazel, left_on="Approved symbol", right_on="Genes"
            )
            overall = overall.merge(
                id_map, how="left", left_on="HGNC_ID_x", right_on="HGNC ID"
            )
            print(overall.columns.values.tolist())

            del hazel

            print("Combining Ditto and Hazel scores....")
            overall["combined_cosine"] = (
                overall["cosine"].fillna(0)
                + overall["Ditto_Deleterious"].fillna(0)
            ) / 2
            overall["combined_projection"] = (
                overall["projection"].fillna(0)
                + overall["Ditto_Deleterious"].fillna(0)
            ) / 2
            overall["combined_jaccard"] = (
                overall["jaccard"].fillna(0)
                + overall["Ditto_Deleterious"].fillna(0)
            ) / 2

            overall = overall[
                [
                    "SYMBOL_x",
                    "Chromosome",
                    "Position",
                    "Reference Allele",
                    "Alternate Allele",
                    "Ditto_Deleterious",
                    "cosine","projection","jaccard",
                    "combined_cosine",
                    "combined_projection",
                    "combined_jaccard",
                ]
            ]

            overall.insert(0, "PROBANDID", samples)

            overall.columns = [
                "PROBANDID",
                "SYMBOL",
                "CHROM",
                "POS",
                "REF",
                "ALT",
                "Ditto",
                "cosine","projection","jaccard",
                "combined_cosine",
                "combined_projection",
                "combined_jaccard",
            ]

            overall = overall.sort_values("Ditto", ascending=False)
            overall = overall.reset_index(drop=True)
            print("Writing 'Hazel_Ditto.csv' file to Ditto directory....")
            overall.to_csv(f"{args.ditto}/train/{samples}/Hazel_Ditto.csv", index=False)

            overall = overall.drop_duplicates(
                subset=["CHROM", "POS", "REF", "ALT"], keep="first"
            ).reset_index(drop=True)
            overall = overall.sort_values("combined_cosine", ascending=False)
            overall.head(100).to_csv(f"{args.ditto}/train/{samples}/Hazel_Ditto_100.csv", index=False)

            del overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=str, required=True, help="Input cohort metadata file with path."
    )
    parser.add_argument(
        "--ditto", type=str, required=True, help="Path to Ditto output directory."
    )
    parser.add_argument(
        "--hazel",
        type=str,
        # default="predictions.csv",
        help="Path to Hazel output directory",
    )
    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.json):
        print("Can't process because the json file ", args.json, " doesn't exist.")
        sys.exit(1)
    if not os.path.isdir(args.hazel):
        print("Can't process because Hazel directory ", args.hazel, " is not a directory.")
        sys.exit(2)
    if not os.path.isdir(args.ditto):
        print("Can't process because Ditto directory ", args.ditto, " is not a directory.")
        sys.exit(2)

    main(args)
