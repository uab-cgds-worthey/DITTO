#python src/analysis/extract_sign_transcripts.py -i data/processed/xzzlthk_DITTO_scores.csv.gz -o data/processed/sign_transcripts.csv

import argparse
from pathlib import Path
import os
import pandas as pd

def check_scores(df):
    df_path = df[df['DITTO'] > 0.9]
    df_benign = df[df['DITTO'] < 0.6]
    if len(df_path) > 0 and len(df_benign) > 0:
        return df

def read_csv_row(file_name, outfile):
    file = pd.read_csv(file_name, on_bad_lines='skip')
    print("File loaded!")
    temp_df = pd.DataFrame(columns = ['transcript','gene','consequence','chrom','pos','ref_base','alt_base','DITTO'] )
    for index, row in file.iterrows():
        if (index < len(file) - 2):
            if (file.loc[index + 1,'chrom'] == row['chrom']) & (file.loc[index + 1,'pos'] == row['pos']) & (file.loc[index + 1,'ref_base'] == row['ref_base']) & (file.loc[index + 1,'alt_base'] == row['alt_base']) :
                temp_df = pd.concat([temp_df, file.iloc[[index+1]], file.iloc[[index]]]).drop_duplicates().reset_index(drop=True)

            else:
                # print(temp_df)
                if len(temp_df) > 0:
                    sign_df = check_scores(temp_df)
                    if sign_df is not None:
                        sign_df.to_csv(outfile, mode='a', header=False, index=False)
                temp_df = temp_df[0:0]


# ENST00000506348,CDHR2,missense_variant,chr5,176584512,T,C,0.5604467
# ENST00000261944,CDHR2,missense_variant,chr5,176584512,T,C,0.9707699
# ENST00000510636,CDHR2,missense_variant,chr5,176584512,T,C,0.79089457


def is_valid_output_file(p, arg):
    if os.access(Path(os.path.expandvars(arg)).parent, os.W_OK):
        return os.path.expandvars(arg)
    else:
        p.error(f"Output file {arg} can't be accessed or is invalid!")


def is_valid_file(p, arg):
    if not Path(os.path.expandvars(arg)).is_file():
        p.error(f"The file {arg} does not exist!")
    else:
        return os.path.expandvars(arg)


if __name__ == "__main__":


    PARSER = argparse.ArgumentParser(
        description="Simple parser for creating data model, data parsing config, and data parsing of annotations from OpenCravat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    PARSER.add_argument(
        "-i",
        "--input_csv",
        help="File path to the CSV file of annotated variants from OpenCravat",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="Output from parsing",
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b",
    )



    ARGS = PARSER.parse_args()

    outfile = ARGS.output if ARGS.output else f"{Path(ARGS.input_csv).stem}.csv"
    read_csv_row(ARGS.input_csv, outfile)
