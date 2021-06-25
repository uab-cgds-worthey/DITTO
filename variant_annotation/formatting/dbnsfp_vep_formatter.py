import gzip
import argparse 
from pathlib import Path
import os

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
        description="Simple parser for converting dbNSFP multivalue per column into single value per column",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    PARSER.add_argument(
        "-i",
        "--input",
        help="File path to the input dbNSFP gzipped TSV file",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b"
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="File path to the desired output file (default is to use input file location and name but with *.tsv extension)",
        required=False,
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b"
    )

    ARGS = PARSER.parse_args()

    inputf = Path(ARGS.input)
    outf = Path(ARGS.output) if ARGS.output else inputf.parent / f"{inputf.stem}.tsv"

    with outf.open('w') as outfp:
        with gzip.open(inputf, 'rt') as fp:
            for _, line in enumerate(fp):
                cols = line.rstrip("\n").split("\t")
                genes = cols[13].split(";")
                trxs = cols[14].split(";")
                gene_len = len(genes)
                if len(genes) != len(trxs) or len(trxs) == 1:
                    outfp.write(line)
                else:
                    split_cols = list()
                    for col in cols:
                        _split_col = col.split(";")
                        if len(_split_col) == gene_len:
                            split_cols.append(_split_col)
                        else:
                            split_cols.append(col)

                    for new_row_num in range(0, gene_len):
                        new_line = list()
                        for new_col in split_cols:
                            if type(new_col) == list:
                                new_line.append(new_col[new_row_num])
                            else:
                                new_line.append(new_col)
                        
                        outfp.write("\t".join(new_line) + "\n")

