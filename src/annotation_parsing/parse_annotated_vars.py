from pathlib import Path
import argparse
import os
import json
import csv

ALL_MAPPINGS_COLUMN_ID = "all_mappings"


def create_data_config(annot_csv, outfile = None):
    #Column description. Column 0 uid=UID
    #Column description. Column 1 chrom=Chrom
    #Column description. Column 2 pos=Position
    #Column description. Column 3 ref_base=Ref Base
    #Column description. Column 4 alt_base=Alt Base
    columns = list()
    with open(annot_csv) as csvfp:
        cntr = 0
        for line in csvfp:
            if line.startswith("#Column description. Column"):
                line = line.replace("#Column description. Column ","").strip()
                info = line.replace("="," ").split(" ")
                columns.append({
                    "col_num": info[0],
                    "col_id": info[1],
                    "description": info[2],
                    "parse": True,
                    "parse_type": {
                        "none":{},
                        "list_index": {
                            "separator": ";"
                        },
                        "list":{
                            "trx_index_col": "fathmm.ens_tid",
                            "separator": ";"
                        },
                        "list-o-dicts":{
                            "dict_index": {
                                0: "column_name",
                                1: "column_name1"
                            },
                            "trx_mapping_col_index": 0
                        }
                    }
                })
            if cntr > 2000:
                break
            else:
                cntr += 1

    with open(outfile, "wt") as ofp:
        json.dump(columns,ofp)


def parse_list_of_dicts(data_value):
    list_of_dicts = list()
    if data_value.startswith("[["):
        # parse list of dicts that uses json formatting
        for sublist in json.loads(data_value):
            sublist_dict = dict()
            list_of_dicts.append(sublist_dict)
            for index, value in enumerate(sublist):
                sublist_dict[index] = value
    else:
        for sublist in data_value.split(";"):
            sublist_dict = dict()
            list_of_dicts.append(sublist_dict)
            for index, value in enumerate(sublist.trim().split(":")):
                sublist_dict[index] = value    


def parse_annotations(annot_csv, data_config_file, outfile):
    # reading data config for determination of parsing
    data_config = list()
    with open(data_config_file, "rt") as dcfp:
        # parse and filter for column configs that needing parsing
        data_config = [filter(lambda colconf: colconf["parse"], json.load(dcfp))]

    # the column "all_mappings" is the key split-by column to separate results on a per variant + transcript
    with open(outfile, "w", newline="") as paserdcsv:
        parse_fieldnames = [colconf["col_id"] for colconf in data_config]
        hardcoded_fieldnames = ["trx", "gene", "consequence", "protein_hgvs", "cdna_hgvs"]
        csvwriter = csv.DictWriter(paserdcsv, fieldnames=hardcoded_fieldnames + parse_fieldnames)
        csvwriter.writeheader()
        with open(annot_csv, "r", newline="") as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile))
            for row in reader:
                for variant_trx in row[ALL_MAPPINGS_COLUMN_ID].split(";"):
                    vtrx_cols = variant_trx.split(":")
                    trx = vtrx_cols[0].split(".")[0]
                    gene = vtrx_cols[1]
                    vtrx_consequence = vtrx_cols[3]
                    protein_hgvs = vtrx_cols[4]
                    cdna_hgvs = vtrx_cols[5]
                    for column in parse_fieldnames:
                        if "list-o-dicts" in column["parse_type"]:
                            parse_list_of_dicts(row[column["col_id"]])
                        elif "list" in column["parse_type"]:
                            row[column["col_id"]].split(column["separator"])
                        elif "list_index" in column["parse_type"]:
                            continue
                        else:
                            row[column["col_id"]]


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
    EXECUTIONS = [
        "config",
        "parse",
    ]

    PARSER = argparse.ArgumentParser(
        description="Simple parser for creating data model, data parsing config, and data parsing of annotations from OpenCravat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    PARSER.add_argument(
        "-i",
        "--input_csv",
        help="File path to the CSV file of annotated variants from OpenCravat",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b"
    )

    PARSER.add_argument(
        "-e",
        "--exec",
        help="Determine what should be done: create a new data config file or parse the annotations from the OpenCravat CSV file",
        required=True,
        choices=EXECUTIONS,
        metavar="\b"
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="Output from parsing",
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b"
    )

    PARSER.add_argument(
        "-v",
        "--version",
        help="Verison of OpenCravat used to generate the config file (only required during config parsing)",
        type=str,
        metavar="\b"
    )

    PARSER.add_argument(
        "-c",
        "--config",
        help="File path to the data config JSON file that determines how to parse annotated variants from OpenCravat",
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b"
    )

    ARGS = PARSER.parse_args()

    if ARGS.exec == "config" and not ARGS.version:
        print("Version of OpenCravat must be specified when creating a config from their data for tracking purposes")
        raise SystemExit(1)

    if ARGS.exec == "config":
        create_data_config(ARGS.input_csv, f"opencravat_{ARGS.version}_config.json")
    else:
        outfile = ARGS.outfile if ARGS.outfile else f"{Path(ARGS.input_csv).stem}.csv"
        parse_annotations(ARGS.input_csv, ARGS.config, outfile)
