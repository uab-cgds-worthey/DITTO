# VEP annotated VCF to TSV parser

This is a simple, no extra fluff parser for taking an annotated VCF produced via VEP and converting it
to a Tab Seperated Values (TSV) file.

## Requirements

 - Python 3.7+
 - VEP annotated VCF file (text or gzipped)

## Input and Output Format Info

 - The parser is agnostic to which VEP fields are present, it pulls the column headers and ordering from VCF info
 - Works on VCF files with no sample info, with 1 sample, and with multiple samples
 - Columns without annotated information for a variant are left blank (i.e. there can be multiple tabs without
 data between them)
 - Certain annotated fields from VEP have information in them seperated by characters like `&` and are **_NOT_**
 parsed by this parser, that is up to downstream users of the parsed information
 - Variants affecting more than 1 transcript will have their variant information duplicated and each transcript
 of info will be printed on its own line (e.g. a variant affects 2 transcripts it will have two rows in the output
 with the same variant info but each transcripts worth of info seperated onto one of those lines)

## How to Run

To parse the example provided with this repo:

```sh
python parse_annotated_vars.py -i ../variant_annotation/.test/data/processed/vep/testing_variants_hg38_vep-annotated.vcf.gz -o .test/testing_variants_hg38_vep-annotated.tsv
```

Or run the parser without any arguments to get more help info.