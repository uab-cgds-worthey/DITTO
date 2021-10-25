from pathlib import Path
import argparse
import os
import gzip


def parse_n_print(vcf, outfile):
    # collect header information for the annotated information as well as the sample itself
    print("Collecting header info...")
    output_header = list()
    vcf_header = list()
    with gzip.open(vcf, 'rt') if vcf.suffix == ".gz" else vcf.open('r') as vcffp:
        for cnt, line in enumerate(vcffp):
            line = line.rstrip("\n")
            if line.startswith("#"):
                if "ID=CSQ" in line:
                    output_header = ["Chromosome", "Position", "Reference Allele", "Alternate Allele"] + \
                        line.replace(" Allele|"," VEP_Allele_Identifier|").split("Format: ")[1].rstrip(">").rstrip('"').split("|")
                elif line.startswith("#CHROM"):
                    vcf_header = line.split("\t")
            else:
                break

    for idx, sample in enumerate(vcf_header):
        if idx > 8:
            output_header.append(f"{sample} allele depth")
            output_header.append(f"{sample} total depth")
            output_header.append(f"{sample} allele percent reads")

    with open(outfile, "w") as out:
        out.write("\t".join(output_header) + "\n")
        print("Parsing variants...")
        with gzip.open(vcf, 'rt') if vcf.suffix == ".gz" else vcf.open('r') as vcffp:
            for cnt, line in enumerate(vcffp):
                if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    csq = parse_csq(next(filter(lambda info: info.startswith("CSQ="),cols[7].split(";"))).replace("CSQ=",""))
                    #print(line, file=open("var_info.txt", "w"))
                    #var_info = parse_var_info(vcf_header, cols)
                    alt_alleles = cols[4].split(",")
                    alt2csq = format_alts_for_csq_lookup(cols[3], alt_alleles)
                    for alt_allele in alt_alleles:
                        possible_alt_allele4lookup = alt2csq[alt_allele]
                        if possible_alt_allele4lookup not in csq.keys():
                            possible_alt_allele4lookup = alt_allele
                        try:
                            write_parsed_variant(
                                out,
                                vcf_header,
                                cols[0],
                                cols[1],
                                cols[3],
                                alt_allele,
                                csq[possible_alt_allele4lookup]
                                #,var_info[alt_allele]
                            )
                        except KeyError:
                            print("Variant annotation matching based on allele failed!")
                            print(line)
                            print(csq)
                            print(alt2csq)
                            raise SystemExit(1)


def write_parsed_variant(out_fp, vcf_header, chr, pos, ref, alt, annots):#, var_info):
    var_list = [chr, pos, ref, alt]
    for annot_info in annots:
        full_fmt_list = var_list + annot_info
        #for idx, sample in enumerate(vcf_header):
        #    if idx > 8:
        #        full_fmt_list.append(str(var_info[sample]["alt_depth"]))
        #        full_fmt_list.append(str(var_info[sample]["total_depth"]))
        #        full_fmt_list.append(str(var_info[sample]["prct_reads"]))

        out_fp.write("\t".join(full_fmt_list) + "\n")


def format_alts_for_csq_lookup(ref, alt_alleles):
    alt2csq = dict()
    dels = list()
    for alt in alt_alleles:
        if len(ref) == len(alt):
            alt2csq[alt] = alt
        elif alt.startswith(ref):
            alt2csq[alt] = alt[1:]
        else:
            dels.append(alt)

    if len(dels) > 0:
        min_length = min([len(alt) for alt in dels])
        for alt in dels:
            if min_length == len(alt):
                alt2csq[alt] = "-"
            else:
                alt2csq[alt] = alt[1:]

    return alt2csq


def parse_csq(csq):
    csq_allele_dict = dict()
    for annot in csq.split(","):
        parsed_annot = annot.split("|")
        if parsed_annot[0] not in csq_allele_dict:
            csq_allele_dict[parsed_annot[0]] = list()

        csq_allele_dict[parsed_annot[0]].append(parsed_annot)

    return csq_allele_dict


def parse_var_info(headers, cols):
    if len(cols) < 9:
        return {alt_allele: dict() for alt_allele in cols[4].split(",")}
    else:
        ad_index = cols[8].split(":").index("AD")
        parsed_alleles = dict()
        for alt_index, alt_allele in enumerate(cols[4].split(",")):
            allele_dict = dict()
            for col_index, col in enumerate(cols):
                if col_index > 8:
                    ad_info = col.split(":")[ad_index]
                    alt_depth = 0
                    total_depth = 0
                    prct_reads = 0
                    sample = headers[col_index]
                    if ad_info != ".":
                        ad_info = ad_info.replace(".", "0").split(",")
                        alt_depth = int(ad_info[alt_index + 1])
                        total_depth = sum([int(dp) for dp in ad_info])
                        prct_reads = (alt_depth / total_depth) * 100

                    allele_dict[sample] = {
                        "alt_depth": alt_depth,
                        "total_depth": total_depth,
                        "prct_reads": prct_reads
                    }

            parsed_alleles[alt_allele] = allele_dict

        return parsed_alleles


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
        "--input_vcf",
        help="File path to the input VEP annotated VCF file to parse",
        required=True,
        type=lambda x: is_valid_file(PARSER, x),
        metavar="\b"
    )

    OPTIONAL_ARGS = PARSER.add_argument_group("Override Args")
    PARSER.add_argument(
        "-o",
        "--output",
        help="File path to the desired output file (default is to use input VCF location and name but with *.tsv extension)",
        required=False,
        type=lambda x: is_valid_output_file(PARSER, x),
        metavar="\b"
    )

    ARGS = PARSER.parse_args()

    inputf = Path(ARGS.input_vcf)
    outputf = Path(ARGS.output) if ARGS.output else inputf.parent / inputf.stem.rstrip(".vcf") + ".tsv"

    parse_n_print(inputf, outputf)
