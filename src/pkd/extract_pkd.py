import gzip
import os

def extract_variants(input, output):
    print(f"Writing PKD variants from {input}...")
    pkd1 = pkd2 = 0
    with gzip.open(output, "wt") as out:
        with gzip.open(input, "rt") as vcffp:
            for cnt, line in enumerate(vcffp):
                if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    if  cols[12] == 'PKD1':#12
                        pkd1 = pkd1 + 1
                        out.write(line + "\n")
                    elif cols[12] == 'PKD2':
                        pkd2 = pkd2 + 1
                        out.write(line + "\n")
                else:
                    out.write(line)

    print(f"PKD1 variants: {pkd1}\nPKD2 variants: {pkd2}\n")
    return None

if __name__ == "__main__":

    extract_variants("/data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.sorted.tsv.gz","/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/dbNSFP_PKD_variants.tsv.gz")
