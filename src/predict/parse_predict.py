from pathlib import Path
import argparse
import os
import json
import csv
import ctypes as ct
import gzip
import pandas as pd
from tqdm import tqdm
from tensorflow import keras
import yaml

# dealing with large fields in a CSV requires more memory allowed per field
# see https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072 for discussion
# and this solution
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

ALL_MAPPINGS_COLUMN_ID = "all_mappings"


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

    for index, index_value in enumerate(index_column.split(multi_column_config["separator"])):
        index_mapping[index] = index_value.split(".")[0]
        dict_of_dicts[index_mapping[index]] = dict()

    for column, value in data_cols_dict.items():
        sublist = value.split(multi_column_config["separator"])
        for index, data_value in enumerate(sublist):
            dict_of_dicts[index_mapping[index]][column] = data_value

    return dict_of_dicts


def test_parsing(dataframe, config_dict, clf):
    # Drop variant info columns so we can perform one-hot encoding
    # dataframe = dataframe[config_dict["raw_cols"]]
    dataframe["so"] = dataframe["consequence"]
    var = dataframe[config_dict["id_cols"]]
    dataframe = dataframe.drop(config_dict["id_cols"], axis=1)
    # dataframe = dataframe.replace(['.','-'], np.nan)

    # Perform one-hot encoding
    for key in config_dict["dummies_sep"]:
        dataframe = pd.concat(
            (dataframe, dataframe[key].str.get_dummies(sep=config_dict["dummies_sep"][key])), axis=1
        )
    dataframe = dataframe.drop(list(config_dict["dummies_sep"].keys()), axis=1)
    dataframe = pd.get_dummies(dataframe, prefix_sep="_")

    df2 = pd.DataFrame(columns=config_dict["filtered_cols"])
    for key in config_dict["filtered_cols"]:
        if key in dataframe.columns:
            df2[key] = dataframe[key]
        else:
            df2[key] = 0
    del dataframe

    df2 = df2.drop(config_dict["train_cols"], axis=1)
    for key in list(config_dict["median_scores"].keys()):
        if key in df2.columns:
            df2[key] = df2[key].fillna(config_dict["median_scores"][key]).astype("float64")

    df2 = df2*1
    y_score = 1 - clf.predict(df2, verbose=0)
    y_score = pd.DataFrame(y_score, columns=["DITTO"])

    var = pd.concat([var.reset_index(drop=True), y_score.reset_index(drop=True)], axis=1)
    return var


def parse_annotations(annot_csv, data_config_file, outfile, clf, config_dict):
    # reading data config for determination of parsing
    data_config = list()
    with open(data_config_file, "rt") as dcfp:
        # parse and filter for column configs that needing parsing
        data_config = json.load(dcfp)

    # the column "all_mappings" is the key split-by column to separate results on a per variant + transcript
    with open(outfile, "w", newline="") as paserdcsv:
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
                parsed_fieldnames += colconf["parse_type"]["list-o-dicts"]["dict_index"].values()
            else:
                parsed_fieldnames.append(colconf["col_id"])

        # Create a set of pre-defined keys
        predefined_keys = set(hardcoded_fieldnames + parsed_fieldnames)
        # csvwriter = csv.DictWriter(paserdcsv, fieldnames=hardcoded_fieldnames + parsed_fieldnames)
        # csvwriter.writeheader()
        pd.DataFrame(columns=config_dict["id_cols"] + ["DITTO"]).to_csv(paserdcsv, index=False)

        with gzip.open(annot_csv, "rt", newline="") if annot_csv.endswith(".gz") else open(
            annot_csv, "r", newline=""
        ) as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0] != "#", csvfile))
            for row in reader:
                df_list = list()
                # parse list of dict columns first since this only needs to be done once per row and cached
                cached_dicts_o_dicts = dict()

                for column in filter(
                    lambda colconf: "list-o-dicts" in colconf["parse_type"], data_config
                ):
                    cached_dicts_o_dicts[column["col_id"]] = parse_list_of_dicts(
                        row[column["col_id"]], column["parse_type"]["list-o-dicts"]
                    )

                # parse list which is a list of dicts spread across multiple columns
                for column in filter(lambda colconf: "list" in colconf["parse_type"], data_config):
                    col_data_dict = {
                        subcolumn: row[subcolumn]
                        for subcolumn in column["parse_type"]["list"]["column_list"]
                    }
                    cached_dicts_o_dicts[column["col_id"]] = parse_multicolumn_list_of_dicts(
                        row[column["col_id"]],
                        column["parse_type"]["list"],
                        col_data_dict,
                    )

                # parse each variant + transcript
                for variant_trx in row[ALL_MAPPINGS_COLUMN_ID].split(";"):
                    vtrx_cols = variant_trx.split(":")
                    trx = vtrx_cols[0].split(".")[0].strip()
                    # Initialize annot_variant with all pre-defined keys set to None
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
                                for subcol in column["parse_type"]["list"]["column_list"]:
                                    annot_variant[subcol] = ""
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
                                annot_variant.update(cached_dicts_o_dicts[column["col_id"]][trx])
                            else:
                                continue

                    # print parsed variant + transcript annotations to csv file output
                    # csvwriter.writerow(annot_variant)

                    df_list.append(annot_variant)

                df = test_parsing(pd.DataFrame(df_list), config_dict, clf)
                df.to_csv(paserdcsv, mode="a", header=False, index=False)
    return None


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

    PARSER.add_argument(
        "-c",
        "--config",
        help="File path to the data config JSON file that determines how to parse annotated variants from OpenCravat",
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b",
    )

    ARGS = PARSER.parse_args()
    clf = keras.models.load_model("model/Neural_network")
    clf.load_weights("model/weights.h5")
    with open("configs/col_config.yaml") as fh:
        config_dict = yaml.safe_load(fh)

    outfile = ARGS.output if ARGS.output else f"{Path(ARGS.input_csv).stem}.csv"
    DExTR_dict = {}  # get_DExTR_dict("DExTR/GEP_scores_transcripts_scaled.csv.gz")
    parse_annotations(ARGS.input_csv, ARGS.config, outfile, clf, config_dict)
