from pathlib import Path
import argparse
import os
import json
import csv


# Example fields for parsing and normalizing:
# 'all_mappings': 'ENST00000253952.9:THOC6:Q86W42:missense_variant:p.Val234Leu:c.700G>C; ENST00000326266.13:THOC6:Q86W42:missense_variant:p.Val234Leu:c.700G>C; ENST00000389347.4:BICDL2:A1A5D9:2kb_downstream_variant::c.*936C>G; ENST00000572449.6:BICDL2:A1A5D9:2kb_downstream_variant::c.*936C>G; ENST00000573514.5:BICDL2:A1A5D9:2kb_downstream_variant::c.*936C>G; ENST00000574549.5:THOC6:Q86W42:missense_variant:p.Val210Leu:c.628G>C; ENST00000575576.5:THOC6:Q86W42:missense_variant:p.Val210Leu:c.628G>C; ENST00000642419.1:BICDL2::2kb_downstream_variant::c.*936C>G'
# 'chasmplus.all': '[["ENST00000574549.5", 0.064, 0.314], ["ENST00000575576.5", 0.064, 0.314], ["NM_001142350.1", 0.055, 0.358], ["NM_024339.3", 0.047, 0.405]]'
# 'biogrid.acts': 'EFTUD2;PPP2R1A;RRP9;SNRNP200;THOC1;THOC7;TPR;TRIM55;U2AF1;U2AF2;UTRN;VDAC2;ZC3H15;ZCCHC8;ZNF326'
# 'clinvar.sig_conf': 'Pathogenic(1)|Likely pathogenic(2)|Uncertain significance(3)'
# 'clinvar.disease_refs': 'MONDO:MONDO:0013362,MedGen:C3150939,OMIM:613680,Orphanet:ORPHA363444|MeSH:D030342,MedGen:C0950123|MedGen:CN517202'
# 'funseq2.all': '[["", "", "", "", "", "", "4"]]'
# 'intact.intact': 'GABARAPL2[20562859]|NUDC[25036637]|JUN[25609649]|THOC1[19165146;26496610]|THOC2[19165146;26496610]|DDX41[25920683]|THOC5[26496610]|ESR2[21182203]|GABARAP[20562859]|THOC7[26496610]|PLEKHA7[28877994]|BCLAF1[26496610]|MAP1LC3A[20562859]|ID1[26496610]|ABI1[26496610]|NCBP3[26496610;26382858]|'

# TODO create config for field mappings and parsing logic needed for various field types from examples above


# list of dictionaries (that looks like a list of lists, can have empty values), this will require mapping configuration


# list


# lists that don't need parsing


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
                    "parse_type": "list-o-dicts,list,none",
                    "separator": ";"
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
        csvwriter = csv.DictWriter(paserdcsv, fieldnames=[colconf["col_id"] for colconf in data_config])
        csvwriter.writeheader()
        with open(annot_csv, "r", newline="") as csvfile:
            reader = csv.DictReader(filter(lambda row: row[0]!='#', csvfile))
            for row in reader:
                for column in data_config:
                    # TODO rewrite parsing and configs to focus on "all_mappings" column
                    if column["parse_type"] == "list-o-dicts":
                        parse_list_of_dicts(row[column["col_id"]])
                    elif column["parse_type"] == "list":
                        row[column["col_id"]].split(column["separator"])
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
