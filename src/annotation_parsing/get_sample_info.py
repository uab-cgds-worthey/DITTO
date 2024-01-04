# Script to extract sample information from the OpenCRAVAT annotated file

import csv
import ctypes as ct

# dealing with large fields in a CSV requires more memory allowed per field
# see https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072 for discussion
# and this solution
csv.field_size_limit(int(ct.c_ulong(-1).value // 2))

input_filename = 'data/ciliopathies_exomes_2569.vcf.gz.variant.csv'
output_samples = 'data/samples.csv'

with open(input_filename, 'r') as infile, open(output_samples, 'w', newline='') as outfile1:
    reader = csv.reader(filter(lambda row: row[0] != "#", infile))
    writer1 = csv.writer(outfile1)

    for row in reader:
        line = row[1] + "," + row[2] + "," + row[3] + "," + row[4] + "," + row[15] + "," + row[16]
        outfile1.write(line + '\n')

print("Columns replaced and output written to", output_samples)
