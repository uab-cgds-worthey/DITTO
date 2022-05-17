#python src/training/data-prep/parse_dbNSFP.py -i /data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.bgz -o /data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.bgz

from pathlib import Path
import argparse
import os
import gzip

def parse_n_print(vcf, outfile):
    with  gzip.open(outfile, "wt") if outfile.suffix == ".gz" else outfile.open('w')as out:
        print("Parsing variants...")
        with gzip.open(vcf, 'rt') if vcf.suffix == ".bgz" else vcf.open('r') as vcffp:
            for cnt, line in enumerate(vcffp):
                if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    transcripts = cols[14].split(";")
                    if len(transcripts) > 1:
                        for idx in range(len(transcripts)):
                            col_list = []
                            for col in cols:
                                if ';' in col:
                                    if len(col.split(';'))==len(transcripts):
                                        col_list.append(col.split(';')[idx])
                                    else:
                                        col_list.append(col)
                                else:
                                    col_list.append(col)
                            out.write("\t".join(col_list) + "\n")
                    else:
                        out.write(line + "\n")
                else:
                    out.write(line)


def is_valid_output_file(p, arg):
    if os.access(Path(os.path.expandvars(arg)).parent, os.W_OK):
        return os.path.expandvars(arg)
    else:
        p.error(f"Output file {arg} can't be accessed or is invalid!")


def is_valid_file(p, arg):
    if not Path(os.path.expandvars(arg)).is_file():
        p.error("The file '%s' does not exist!" % arg)
    else:
        return os.path.expandvars(arg)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        description="Simple parser for converting an annotated VCF file produced by VEP into a columnar format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    PARSER.add_argument(
        "-i",
        "--input",
        help="File path to the input dbNSFP file to parse",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b"
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="File path to the desired output file (default is to use input file location and name but with *.tsv.gz extension)",
        required=False,
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b"
    )

    ARGS = PARSER.parse_args()

    inputf = Path(ARGS.input)
    outputf = Path(ARGS.output) if ARGS.output else inputf.parent / inputf.stem.rstrip(".bgz") + ".tsv"

    parse_n_print(inputf, outputf)
