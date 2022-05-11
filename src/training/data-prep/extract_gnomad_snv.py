import gzip

input = "/data/project/worthey_lab/temp_datasets_central/mana/gnomad/v3.0/data/gnomad.genomes.r3.0.sites.vcf.bgz"
output = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/interim/gnomad_snv.vcf"

print("Writing gnomad SNV variants...")
with open(output, "w") as out:
    with gzip.open(input, "rt") as vcffp:
        for cnt, line in enumerate(vcffp):
            if not line.startswith("#"):
                line = line.rstrip("\n")
                cols = line.split("\t")
                if len(cols[3]) <2 and len(cols[4]) <2 and cols[7].split(';')[4] == 'variant_type=snv':
                    out.write(line + "\n")
                # print(line+"\t"+class_dict[var_info])
            else:
                out.write(line)
