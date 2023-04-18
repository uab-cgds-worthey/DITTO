from pathlib import Path
import argparse
import os
import json


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
                    "parse": "true|false",
                    "parse_type": "list-o-dicts,list,none"
                })
            if cntr > 2000:
                break
            else:
                cntr += 1

    with open(outfile, "wt") as ofp:
        json.dump(columns,ofp)


# def parse_n_print(vcf, outfile):
#     # collect header information for the annotated information as well as the sample itself
#     print("Collecting header info...")
#     output_header = list()
#     vcf_header = list()
#     with gzip.open(vcf, 'rt') if vcf.suffix == ".gz" else vcf.open('r') as vcffp:
#         for cnt, line in enumerate(vcffp):
#             line = line.rstrip("\n")
#             if line.startswith("#"):
#                 if "ID=CSQ" in line:
#                     output_header = ["Chromosome", "Position", "Reference Allele", "Alternate Allele"] + \
#                         line.replace(" Allele|"," VEP_Allele_Identifier|").split("Format: ")[1].rstrip(">").rstrip('"').split("|")
#                 elif line.startswith("#CHROM"):
#                     vcf_header = line.split("\t")
#             else:
#                 break

#     for idx, sample in enumerate(vcf_header):
#         if idx > 8:
#             output_header.append(f"{sample} allele depth")
#             output_header.append(f"{sample} total depth")
#             output_header.append(f"{sample} allele percent reads")

#     with open(outfile, "w") as out:
#         out.write("\t".join(output_header) + "\n")
#         print("Parsing variants...")
#         with gzip.open(vcf, 'rt') if vcf.suffix == ".gz" else vcf.open('r') as vcffp:
#             for cnt, line in enumerate(vcffp):
#                 if not line.startswith("#"):
#                     line = line.rstrip("\n")
#                     cols = line.split("\t")
#                     csq = parse_csq(next(filter(lambda info: info.startswith("CSQ="),cols[7].split(";"))).replace("CSQ=",""))
#                     #print(line, file=open("var_info.txt", "w"))
#                     #var_info = parse_var_info(vcf_header, cols)
#                     alt_alleles = cols[4].split(",")
#                     alt2csq = format_alts_for_csq_lookup(cols[3], alt_alleles)
#                     for alt_allele in alt_alleles:
#                         possible_alt_allele4lookup = alt2csq[alt_allele]
#                         if possible_alt_allele4lookup not in csq.keys():
#                             possible_alt_allele4lookup = alt_allele
#                         try:
#                             write_parsed_variant(
#                                 out,
#                                 vcf_header,
#                                 cols[0],
#                                 cols[1],
#                                 cols[3],
#                                 alt_allele,
#                                 csq[possible_alt_allele4lookup]
#                                 #,var_info[alt_allele]
#                             )
#                         except KeyError:
#                             print("Variant annotation matching based on allele failed!")
#                             print(line)
#                             print(csq)
#                             print(alt2csq)
#                             raise SystemExit(1)


# def write_parsed_variant(out_fp, vcf_header, chr, pos, ref, alt, annots):#, var_info):
#     var_list = [chr, pos, ref, alt]
#     for annot_info in annots:
#         full_fmt_list = var_list + annot_info
#         #for idx, sample in enumerate(vcf_header):
#         #    if idx > 8:
#         #        full_fmt_list.append(str(var_info[sample]["alt_depth"]))
#         #        full_fmt_list.append(str(var_info[sample]["total_depth"]))
#         #        full_fmt_list.append(str(var_info[sample]["prct_reads"]))

#         out_fp.write("\t".join(full_fmt_list) + "\n")


# def format_alts_for_csq_lookup(ref, alt_alleles):
#     alt2csq = dict()
#     dels = list()
#     for alt in alt_alleles:
#         if len(ref) == len(alt):
#             alt2csq[alt] = alt
#         elif alt.startswith(ref):
#             alt2csq[alt] = alt[1:]
#         else:
#             dels.append(alt)

#     if len(dels) > 0:
#         min_length = min([len(alt) for alt in dels])
#         for alt in dels:
#             if min_length == len(alt):
#                 alt2csq[alt] = "-"
#             else:
#                 alt2csq[alt] = alt[1:]

#     return alt2csq


# def parse_csq(csq):
#     csq_allele_dict = dict()
#     for annot in csq.split(","):
#         parsed_annot = annot.split("|")
#         if parsed_annot[0] not in csq_allele_dict:
#             csq_allele_dict[parsed_annot[0]] = list()

#         csq_allele_dict[parsed_annot[0]].append(parsed_annot)

#     return csq_allele_dict


# def parse_var_info(headers, cols):
#     if len(cols) < 9:
#         return {alt_allele: dict() for alt_allele in cols[4].split(",")}
#     else:
#         ad_index = cols[8].split(":").index("AD")
#         parsed_alleles = dict()
#         for alt_index, alt_allele in enumerate(cols[4].split(",")):
#             allele_dict = dict()
#             for col_index, col in enumerate(cols):
#                 if col_index > 8:
#                     ad_info = col.split(":")[ad_index]
#                     alt_depth = 0
#                     total_depth = 0
#                     prct_reads = 0
#                     sample = headers[col_index]
#                     if ad_info != ".":
#                         ad_info = ad_info.replace(".", "0").split(",")
#                         alt_depth = int(ad_info[alt_index + 1])
#                         total_depth = sum([int(dp) for dp in ad_info])
#                         prct_reads = (alt_depth / total_depth) * 100

#                     allele_dict[sample] = {
#                         "alt_depth": alt_depth,
#                         "total_depth": total_depth,
#                         "prct_reads": prct_reads
#                     }

#             parsed_alleles[alt_allele] = allele_dict

#         return parsed_alleles


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
        required=False,
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b"
    )

    PARSER.add_argument(
        "-v",
        "--version",
        help="Verison of OpenCravat used to generate the config file (only required during config parsing)",
        required=False,
        type=str,
        metavar="\b"
    )

    ARGS = PARSER.parse_args()

    if ARGS.exec == "config" and not ARGS.version:
        print("Version of OpenCravat must be specified when creating a config from their data for tracking purposes")
        raise SystemExit(1)

    if ARGS.exec == "config":
        create_data_config(ARGS.input_csv, f"opencravat_{ARGS.version}_config.json")
    else:
        #  TODO parsing method lolz
        print("TODO")
    
