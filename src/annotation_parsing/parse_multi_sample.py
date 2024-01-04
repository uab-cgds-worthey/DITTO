from pathlib import Path
import argparse
import os
import json
import csv
import ctypes as ct
import gzip

# dealing with large fields in a CSV requires more memory allowed per field
# see https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072 for discussion
# and this solution
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

ALL_MAPPINGS_COLUMN_ID = "all_mappings"
NUMBER_OF_SAMPLES_COLUMN_ID = "numsample"
SAMPLES_COLUMN_ID = "samples"


def create_data_config(annot_csv, outfile=None):
    # Column description. Column 0 uid=UID
    # Column description. Column 1 chrom=Chrom
    # Column description. Column 2 pos=Position
    # Column description. Column 3 ref_base=Ref Base
    # Column description. Column 4 alt_base=Alt Base
    columns = list()
    with open(annot_csv) as csvfp:
        cntr = 0
        for line in csvfp:
            if line.startswith("#Column description. Column"):
                line = line.replace("#Column description. Column ", "").strip()
                info = line.replace("=", " ").split(" ")
                columns.append(
                    {
                        "col_id": info[1],
                        "parse_type": {
                            "none": "none",
                            "list": {
                                "trx_index_col": "fathmm.ens_tid",
                                "column_list": [
                                    "fathmm.fathmm_score",
                                    "fathmm.fathmm_pred",
                                ],
                                "separator": ";",
                            },
                            "list-o-dicts": {
                                "dict_index": {0: "column_name", 1: "column_name1"},
                                "trx_mapping_col_index": 0,
                            },
                        },
                    }
                )
            if cntr > 2000:
                break
            else:
                cntr += 1

    with open(outfile, "wt") as ofp:
        json.dump(columns, ofp)


def parse_list_of_dicts(data_value, column_config):
    dict_of_dicts = dict()
    if data_value == "":
        return dict_of_dicts

    for sublist in json.loads(data_value):
        sublist_dict = dict()
        trx_id = sublist[column_config["trx_mapping_col_index"]].split(".")[0]
        dict_of_dicts[trx_id] = sublist_dict
        for index, value in enumerate(sublist):
            # look up column name by index value, assign column name as key in return dict
            if str(index) not in column_config["dict_index"]:
                continue

            sublist_dict[column_config["dict_index"][str(index)]] = value

    return dict_of_dicts


def parse_multicolumn_list_of_dicts(index_column, multi_column_config, data_cols_dict):
    # data_cols_n_configs => list of tuples where tuple[0] is column value, tuple[1] is column config
    index_mapping = dict()
    dict_of_dicts = dict()

    if index_column == "":
        return dict_of_dicts

    for index, index_value in enumerate(
        index_column.split(multi_column_config["separator"])
    ):
        index_mapping[index] = index_value.split(".")[0]
        dict_of_dicts[index_mapping[index]] = dict()

    for column, value in data_cols_dict.items():
        sublist = value.split(multi_column_config["separator"])
        for index, data_value in enumerate(sublist):
            dict_of_dicts[index_mapping[index]][column] = data_value

    return dict_of_dicts


def parse_annotations(annot_csv, data_config_file, outdir):
    # reading data config for determination of parsing
    data_config = list()
    with open(data_config_file, "rt") as dcfp:
        # parse and filter for column configs that needing parsing
        data_config = json.load(dcfp)

    # reading the number of samples to parse
    # situations where the GT column for sample info is 0/0 or 0|0 will cause OpenCravat to not print
    # the variant information to the output (it put the variant in the error output file instead) but
    # this issue can't be accounted for in the parsing logic here
    samples = set()
    with open(annot_csv, "r", newline="") as csvfile:
        reader = csv.DictReader(filter(lambda row: row[0] != "#", csvfile))
        for row in reader:
            if int(row[NUMBER_OF_SAMPLES_COLUMN_ID]) == 0:
                samples.add(
                    "_"
                )  # use underscore to represent single sample VCF as a convention
                break

            samples.update(row[SAMPLES_COLUMN_ID].split(";"))

    samples = list(samples)

    # create chunks of samples to process to limit number of open file handles and avoid OS open file handle errors
    sample_groups = []
    for group_index_start in range(0, len(samples), 2):
        end = (
            group_index_start + 2
            if (group_index_start + 2) > len(samples)
            else len(samples)
        )
        sample_groups.append(samples[group_index_start:end])

    # the column "all_mappings" is the key split-by column to separate results on a per variant + transcript
    hardcoded_fieldnames = [
        "transcript",
        "gene",
        "consequence",
        "protein_hgvs",
        "cdna_hgvs",
    ]
    parsed_fieldnames = list()
    for colconf in data_config:
        if "list" in colconf["parse_type"]:
            parsed_fieldnames += colconf["parse_type"]["list"]["column_list"]
        elif "list-o-dicts" in colconf["parse_type"]:
            parsed_fieldnames += colconf["parse_type"]["list-o-dicts"][
                "dict_index"
            ].values()
        else:
            parsed_fieldnames.append(colconf["col_id"])

    predefined_keys = hardcoded_fieldnames + parsed_fieldnames

    for sample_group in sample_groups:
        # create dict of output writers for all samples in the group
        sample_output_writers = dict()
        sample_group_fh = []
        for sample in sample_group:
            # with open(outdir, "w", newline="") as paserdcsv:
            samplepath = (
                outdir / f"{sample}_parsed.csv.gz"
                if sample != "_"
                else outdir / f"{Path(annot_csv).stem}_parsed.csv.gz"
            )
            sample_fh = gzip.open(samplepath, "wt", newline="")
            sample_group_fh.append(sample_fh)
            csvwriter = csv.DictWriter(sample_fh, fieldnames=predefined_keys)
            sample_output_writers[sample] = csvwriter
            csvwriter.writeheader()

        with open(annot_csv, "r", newline="") as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0] != "#", csvfile))
            for row in reader:
                # parse list of dict columns first since this only needs to be done once per row and cached
                cached_dicts_o_dicts = dict()

                for column in filter(
                    lambda colconf: "list-o-dicts" in colconf["parse_type"], data_config
                ):
                    cached_dicts_o_dicts[column["col_id"]] = parse_list_of_dicts(
                        row[column["col_id"]], column["parse_type"]["list-o-dicts"]
                    )

                # parse list which is a list of dicts spread across multiple columns
                for column in filter(
                    lambda colconf: "list" in colconf["parse_type"], data_config
                ):
                    col_data_dict = {
                        subcolumn: row[subcolumn]
                        for subcolumn in column["parse_type"]["list"]["column_list"]
                    }
                    cached_dicts_o_dicts[
                        column["col_id"]
                    ] = parse_multicolumn_list_of_dicts(
                        row[column["col_id"]],
                        column["parse_type"]["list"],
                        col_data_dict,
                    )

                for variant_trx in row[ALL_MAPPINGS_COLUMN_ID].split(";"):
                    vtrx_cols = variant_trx.split(":")
                    trx = vtrx_cols[0].split(".")[0].strip()
                    annot_variant = {key: None for key in predefined_keys}
                    if len(vtrx_cols) < 6:
                        # parse intergenic variant
                        annot_variant["transcript"] = ""
                        annot_variant["gene"] = ""
                        annot_variant["consequence"] = ""
                        annot_variant["protein_hgvs"] = ""
                        annot_variant["cdna_hgvs"] = ""
                        for column in data_config:
                            if "none" in column["parse_type"]:
                                annot_variant[column["col_id"]] = row[column["col_id"]]
                            elif "list-o-dicts" in column["parse_type"]:
                                for subcol in column["parse_type"]["list-o-dicts"][
                                    "dict_index"
                                ].values():
                                    annot_variant[subcol] = row[subcol]
                            else:
                                for subcol in column["parse_type"]["list"][
                                    "column_list"
                                ]:
                                    annot_variant[subcol] = row[subcol]
                    else:
                        # parse variant with transcript info
                        annot_variant["transcript"] = trx
                        annot_variant["gene"] = vtrx_cols[1]
                        annot_variant["consequence"] = vtrx_cols[3]
                        annot_variant["protein_hgvs"] = vtrx_cols[4]
                        annot_variant["cdna_hgvs"] = vtrx_cols[5]
                        for column in data_config:
                            if "none" in column["parse_type"]:
                                annot_variant[column["col_id"]] = row[column["col_id"]]
                            elif trx in cached_dicts_o_dicts[column["col_id"]]:
                                annot_variant.update(
                                    cached_dicts_o_dicts[column["col_id"]][trx]
                                )
                            else:
                                continue

                    # print parsed variant + transcript annotations to csv file output
                    samples = row[SAMPLES_COLUMN_ID].split(";")
                    if len(samples) == 1 and samples[0] == "":
                        sample_output_writers["_"].writerow(annot_variant)
                    else:
                        for sample in samples:
                            if sample in sample_output_writers:
                                sample_output_writers[sample].writerow(annot_variant)

        # don't forget to close all the open writer file handles for the group ;)
        for sample_fh in sample_group_fh:
            sample_fh.close()


def is_valid_output_dir(p, arg):
    if os.access(Path(os.path.expandvars(arg)), os.W_OK):
        return os.path.expandvars(arg)
    else:
        p.error(f"Output directory {arg} can't be accessed or is invalid!")


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

    PARSER.add_argument(
        "-e",
        "--exec",
        help="Determine what should be done: create a new data config file or parse the annotations from the OpenCravat CSV file",
        required=True,
        choices=EXECUTIONS,
        metavar="\b",
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="Output directory for parsing",
        type=lambda x: is_valid_output_dir(PARSER, x),
        metavar="\b",
    )

    PARSER.add_argument(
        "-v",
        "--version",
        help="Verison of OpenCravat used to generate the config file (only required during config parsing)",
        type=str,
        metavar="\b",
    )

    PARSER.add_argument(
        "-c",
        "--config",
        help="File path to the data config JSON file that determines how to parse annotated variants from OpenCravat",
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )

    ARGS = PARSER.parse_args()

    if ARGS.exec == "config" and not ARGS.version:
        print(
            "Version of OpenCravat must be specified when creating a config from their data for tracking purposes"
        )
        raise SystemExit(1)

    if ARGS.exec == "config":
        create_data_config(ARGS.input_csv, f"opencravat_{ARGS.version}_config.json")
    else:
        outdir = Path(ARGS.output) if ARGS.output else Path(ARGS.input_csv).parent
        parse_annotations(ARGS.input_csv, ARGS.config, outdir)
