# python src/cohort/combine_scores.py --json /data/project/worthey_lab/projects/experimental_pipelines/mana/small_tasks/cagi6/rgp/data/processed/metadata/train_test_metadata_original.json --ditto /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/debugged/annotated_vcf --hazel /data/project/worthey_lab/projects/experimental_pipelines/tarun/uab-meter/data/processed/CAGI6

import pandas as pd
import warnings
import json
warnings.simplefilter("ignore")
import argparse
import os
import sys

def main(args):

    print("Loading Biomart file....")
    id_map = pd.read_csv(
                "/data/project/worthey_lab/temp_datasets_central/tarun/HGNC/biomart_9_23_21.txt",
                sep="\t",
            )

    print("Loading Ditto file....")
    ditto = pd.read_csv(args.ditto)
    print("Loading Raw file....")
    raw = pd.read_csv(
        args.raw,
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
    hazel = pd.read_csv(args.hazel)

    id_map = id_map.merge(
        hazel, left_on="Approved symbol", right_on="Genes"
    )

    overall = overall.merge(
        id_map, how="left", left_on="HGNC_ID_x", right_on="HGNC ID"
    )

    #print(overall.columns.values.tolist())

    del id_map, hazel

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
    overall.insert(0, "PROBANDID", args.sample)
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
    overall.to_csv(args.output, index=False)

    overall = overall.drop_duplicates(
        subset=["CHROM", "POS", "REF", "ALT"], keep="first"
    ).reset_index(drop=True)

    overall = overall.sort_values("combined_cosine", ascending=False)

    overall.head(100).to_csv(args.output100, index=False)

    del overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=str, required=True, help="Input raw annotated file with path."
    )
    parser.add_argument(
        "--ditto", type=str, required=True, help="Input Ditto file with path."
    )
    parser.add_argument(
        "--hazel",
        type=str,
        # default="predictions.csv",
        help="Input hazel file with path",
    )
    parser.add_argument(
        "--sample",
        type=str,
        # required=True,
        help="Input sample name to showup in results",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="Ditto_Hazel.csv",
        help="Output csv file with path",
    )
    parser.add_argument(
        "--output100",
        "-o100",
        type=str,
        default="Ditto_Hazel_100.csv",
        help="Output csv file with path for Top 100 variants",
    )
    args = parser.parse_args()

    # Validate paths exist
    if not os.path.exists(args.raw):
        print("Can't process because the raw file ", args.raw, " doesn't exist.")
        sys.exit(1)
    if not os.path.exists(args.hazel):
        print("Can't process because Hazel file ", args.hazel, " doesn't exist.")
        sys.exit(1)
    if not os.path.exists(args.ditto):
        print("Can't process because Ditto file ", args.ditto, " doesn't exist.")
        sys.exit(1)

    main(args)
