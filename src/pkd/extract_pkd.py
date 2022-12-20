import gzip
import gc
import os
import ray
# Start Ray.
ray.init(ignore_reinit_error=True)

@ray.remote  # (num_cpus=9)
def extract_variants(input, output, gene):
    print(f"Writing {gene} variants from {input} to {output}...")
    if not os.path.exists(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/{gene}"):
        os.makedirs(f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/{gene}")
    pkd1 = 0
    with gzip.open(output, "wt") as out:
        with gzip.open(input, "rt") as vcffp:
            for cnt, line in enumerate(vcffp):
                if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    if  cols[12] == gene:#12
                        pkd1 = pkd1 + 1
                        out.write(line + "\n")
                    #elif cols[12] == 'PKD2':
                    #    pkd2 = pkd2 + 1
                    #    out.write(line + "\n")
                else:
                    out.write(line)

    print(f"{gene} variants: {pkd1}\n")#PKD2 variants: {pkd2}\n")
    return None

if __name__ == "__main__":

    gene_list = ["LRPPRC","CALM1", "TBC1D4", "MYL6", "STOM", "IQGAP2", "RAC1", "MYL6B", "EIF5B", "IQGAP1", "COX6C", "IQGAP3"]
    remote_ml = [
        extract_variants.remote("/data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.sorted.tsv.gz",f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/{gene}/dbNSFP_{gene}_variants.tsv.gz", gene)

        for gene in gene_list
    ]
    ray.get(remote_ml)
    gc.collect()

