import gzip

input = "/data/project/worthey_lab/temp_datasets_central/tarun/dbNSFP/v4.3_20220319/dbNSFP4.3a_variant.complete.bgz"
output = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/interim/dbNSFP_clinvar_variants.tsv.gz"

clinvar = {}

print("Writing dbNSFP clinvar variants...")
with gzip.open(output, "wt") as out:
    with gzip.open(input, "rt") as vcffp:
        for cnt, line in enumerate(vcffp):
            if not line.startswith("#"):
                line = line.rstrip("\n")
                cols = line.split("\t")
                if cols[631] != ".":
                    #print(cols[631])
                    out.write(line + "\n")
                    if cols[631] not in clinvar.keys():
                        clinvar[cols[631]] = 0
                    else:
                        clinvar[cols[631]] = clinvar[cols[631]]+1
            else:
                out.write(line)

print(clinvar)
