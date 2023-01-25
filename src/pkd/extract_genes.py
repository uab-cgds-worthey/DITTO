import gzip
import gc
import os

def extract_variants(input):
    print(f"Writing variants from {input} ...")

    #with gzip.open(output, "wt") as out:
    with gzip.open(input, "rt") as vcffp:
            for cnt, line in enumerate(vcffp):
                if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    gene = cols[12]
                    if not os.path.exists(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/dbnsfp_genes/{gene}"):
                        os.makedirs(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/dbnsfp_genes/{gene}")

                    with gzip.open(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/dbnsfp_genes/{gene}/dbNSFP_{gene}_variants.tsv.gz", "wt") as out:
                        out.write(line + "\n")

    return None

if __name__ == "__main__":

    extract_variants("/data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.sorted.tsv.gz")

    gc.collect()

