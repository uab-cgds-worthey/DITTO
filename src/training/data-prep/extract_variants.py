
import os
import gzip

os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/external/')
vcf = 'merged.vcf'

with open("../../configs/columns_config.yaml") as fh:
    config_dict = yaml.safe_load(fh)


print("Collecting variant class...")
class_dict = dict()
with gzip.open(vcf, 'rt') as vcffp:
    for cnt, line in enumerate(vcffp):
        if not line.startswith("#"):
                    line = line.rstrip("\n")
                    cols = line.split("\t")
                    var_info = cols[0]+"\t"+cols[1]+"\t"+cols[2]+"\t"+cols[3]+"\t"+cols[4]
                    hgmd_class = cols[7].split(";")[0].split("=")[1]
                    class_dict[var_info] = hgmd_class

#print(class_dict)
print("Writing variant class...")
vcf1 = 'hgmd_pro_2020.4_hg38_edited_vep-annotated.tsv'
with open('hgmd_vep-annotated.tsv', "w") as out:
    with open(vcf1, 'rt') as vcffp:
        for cnt, line in enumerate(vcffp):
            if not line.startswith("Chromosome"):
                        line = line.rstrip("\n")
                        cols = line.split("\t")
                        var_info = cols[0]+"\t"+cols[1]+"\t"+cols[2]+"\t"+cols[3]
                        new_line = line+"\t"+class_dict[var_info]
                        out.write(new_line + "\n")
                        #print(line+"\t"+class_dict[var_info])
            else:
                out.write(line.rstrip("\n") + "\thgmd_class\n")
