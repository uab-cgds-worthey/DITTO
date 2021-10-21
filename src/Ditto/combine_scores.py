import pandas as pd
import warnings

warnings.simplefilter("ignore")
import argparse
import os
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=str, required=True, help="Input raw annotated file with path."
    )
    parser.add_argument(
        "--ditto", type=str, required=True, help="Input Ditto file with path."
    )
    parser.add_argument(
        "--exomiser",
        "-ep",
        type=str,
        # default="predictions.csv",
        help="Path to Exomiser output directory",
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
        default="predictions_with_exomiser.csv",
        help="Output csv file with path",
    )
    parser.add_argument(
        "--output100",
        "-o100",
        type=str,
        default="predictions_with_exomiser_100.csv",
        help="Output csv file with path for Top 100 variants",
    )
    parser.add_argument(
        "--output1000",
        "-o1000",
        type=str,
        default="predictions_with_exomiser_1000.csv",
        help="Output csv file with path for Top 1000 variants",
    )
    args = parser.parse_args()
    # print (args)

    ditto = pd.read_csv(args.ditto)
    raw = pd.read_csv(
        args.raw,
        sep="\t",
        usecols=[
            "SYMBOL",
            "Chromosome",
            "Position",
            "Reference Allele",
            "Alternate Allele",
            "SYMBOL",
            "Gene",
            "Feature",
            "HGNC_ID",
        ],
    )
    # raw = raw[['Chromosome','Position','Reference Allele','Alternate Allele','SYMBOL','Gene','Feature', 'HGNC_ID']]
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
    id_map = pd.read_csv(
        "/data/project/worthey_lab/temp_datasets_central/tarun/HGNC/biomart_9_23_21.txt",
        sep="\t",
    )

    if args.exomiser:
        print("Reading Exomiser scores...")
        all_files = glob.glob(os.path.join(args.exomiser, "*.tsv"))
        exo_scores = pd.concat(
            (pd.read_csv(f, sep="\t") for f in all_files), ignore_index=True
        )
        exo_scores = exo_scores[
            ["#GENE_SYMBOL", "ENTREZ_GENE_ID", "EXOMISER_GENE_PHENO_SCORE"]
        ]
        id_map = id_map.merge(
            exo_scores, left_on="NCBI gene ID", right_on="ENTREZ_GENE_ID"
        )
        overall = overall.merge(
            id_map, how="left", left_on="HGNC_ID_x", right_on="HGNC ID"
        )
        del id_map, exo_scores
        # overall = overall.sort_values(by = ['Ditto_Deleterious','EXOMISER_GENE_PHENO_SCORE'], axis=0, ascending=[False,False], kind='quicksort', ignore_index=True)
        # overall['Exo_norm'] = (overall['EXOMISER_GENE_PHENO_SCORE'] - overall['EXOMISER_GENE_PHENO_SCORE'].min()) / (overall['EXOMISER_GENE_PHENO_SCORE'].max() - overall['EXOMISER_GENE_PHENO_SCORE'].min())
        overall["combined"] = (
            overall["EXOMISER_GENE_PHENO_SCORE"].fillna(0)
            + overall["Ditto_Deleterious"].fillna(0)
        ) / 2
        overall = overall[
            [
                "SYMBOL_x",
                "Chromosome",
                "Position",
                "Reference Allele",
                "Alternate Allele",
                "EXOMISER_GENE_PHENO_SCORE",
                "Ditto_Deleterious",
                "combined",
                "SD",
                "C",
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
            "E",
            "D",
            "P",
            "SD",
            "C",
        ]
        # genes = genes[genes['EXOMISER_GENE_PHENO_SCORE'] != 0]

    # overall.sort_values('pred_Benign', ascending=False).head(500).to_csv(args.output500, index=False)
    else:
        # overall = overall.sort_values('Ditto_Deleterious', ascending=False)
        overall = overall[
            [
                "SYMBOL_x",
                "Chromosome",
                "Position",
                "Reference Allele",
                "Alternate Allele",
                "Ditto_Deleterious",
                "SD",
                "C",
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
            "P",
            "SD",
            "C",
        ]

    overall = overall.sort_values("P", ascending=False)
    overall = overall.reset_index(drop=True)
    overall["SD"] = 0
    overall["C"] = "*"
    overall.to_csv(args.output, index=False)

    overall = overall.drop_duplicates(
        subset=["CHROM", "POS", "REF", "ALT"], keep="first"
    ).reset_index(drop=True)
    overall = overall[["PROBANDID", "CHROM", "POS", "REF", "ALT", "P", "SD", "C"]]
    overall.head(100).to_csv(args.output100, index=False, sep=":")
    overall.head(1000).to_csv(args.output1000, index=False, sep=":")

    # del genes, overall
