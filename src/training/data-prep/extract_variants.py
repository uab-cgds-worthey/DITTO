
import os
import gzip
import yaml
import re
regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')

os.chdir( '/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/external/')
vcf = 'merged.vcf.gz'
output = '../interim/merged_sig.vcf'

with open("../../configs/columns_config.yaml") as fh:
    config_dict = yaml.safe_load(fh)


print("Collecting variant class...")
with open(output, "w") as out:
    with gzip.open(vcf, 'rt') as vcffp:  #gzip.
        for cnt, line in enumerate(vcffp):
            if not line.startswith("#"):
                        line = line.rstrip("\n")
                        cols = line.split("\t")
                        if (len(cols[3]) < 1000) and (len(cols[4]) < 1000) and (regex.search(cols[3]) == None) and (regex.search(cols[4]) == None):
                            var_info = cols[0]+"\t"+cols[1]+"\t"+cols[2]+"\t"+cols[3]+"\t"+cols[4]
                            if 'CLASS' in cols[7]:
                                var_class = cols[7].split(";")[0].split("=")[1]
                                if (var_class in config_dict['ClinicalSignificance']):
                                    new_line = var_info + "\t" + var_class
                                    out.write(new_line + "\n")
                            elif 'CLNSIG' in line:
                                var_class = cols[7].split(";CLN")[5].split("=")[1]
                                var_sub = cols[7].split(";CLN")[4].split("=")[1]
                                if (var_class in config_dict['ClinicalSignificance']) and (var_sub in config_dict['CLNREVSTAT']):
                                    new_line = var_info + "\t" + var_class
                                    out.write(new_line + "\n")
                                    #class_dict[var_info] = var_class
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
            else:
                out.write(line)





#print("Writing variant class...")
#vcf1 = 'hgmd_pro_2020.4_hg38_edited_vep-annotated.tsv'
#with open('hgmd_vep-annotated.tsv', "w") as out:
#    with open(vcf1, 'rt') as vcffp:
#        for cnt, line in enumerate(vcffp):
#            if not line.startswith("Chromosome"):
#                        line = line.rstrip("\n")
#                        cols = line.split("\t")
#                        var_info = cols[0]+"\t"+cols[1]+"\t"+cols[2]+"\t"+cols[3]
#                        new_line = line+"\t"+class_dict[var_info]
#                        out.write(new_line + "\n")
#                        #print(line+"\t"+class_dict[var_info])
#            else:
#                out.write(line.rstrip("\n") + "\thgmd_class\n")
