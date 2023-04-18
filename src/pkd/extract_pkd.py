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

    gene_list = ["PIBF1","KIAA0753","AHI1","ATXN10","B9D1","BCAR1","CC2D2A","CCP110","CEP97","DCTN1","DCTN2","IFT88","INVS","KIF3A","MKS1","NPHP3","PIBF1","KIAA0753","PCNT","PDE6D","RPGR","RPGRIP1","RPGRIP1L","TMEM216","TMEM67","UNC119B", "OFD1", "CEP89","CEP164","CC2D2A", "B9D2", "TCTN1"]
    remote_ml = [
        extract_variants.remote("/data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.parsed.sorted.tsv.gz",f"/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/processed/Ditto/{gene}/dbNSFP_{gene}_variants.tsv.gz", gene)

        for gene in gene_list
    ]
    ray.get(remote_ml)
    gc.collect()

