#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#python src/Ditto/predict.py -i

import pandas as pd
import warnings
warnings.simplefilter("ignore")
import argparse
import os
import io
import gzip
import functools
print = functools.partial(print, flush=True)

os.chdir('/data/project/worthey_lab/temp_datasets_central/tarun/UDN')

def main(args):

    print("Loading Ditto predictions....")

    ditto = pd.read_csv('../Ditto/dbnsfp_only_ditto_predictions.csv.gz')
    ditto = ditto.dropna(subset=['pos(1-based)', 'Ditto_Deleterious'])
    ditto = ditto.sort_values("Ditto_Deleterious", ascending=False)
    ditto = ditto.drop_duplicates(subset=['#chr','pos(1-based)','ref','alt'], keep='first')
    ditto['#chr'] = ditto['#chr'].astype(str)
    ditto['pos(1-based)'] = ditto['pos(1-based)'].astype(int)
    ditto['ref'] = ditto['ref'].astype(str)
    ditto['alt'] = ditto['alt'].astype(str)
    print('Ditto Loaded!\nRunning predictions.....')

    overall = pd.concat([read_vcf('./splits/'+f+'/'+ args.input) for f in os.listdir("./splits/")])
    overall = overall.drop([args.input.split('.')[0],'FILTER','INFO','FORMAT'], axis=1)


    overall = overall.merge(ditto, left_on=['CHROM','POS','REF','ALT'], right_on = ['#chr','pos(1-based)','ref','alt'], how='left')
    del ditto
    overall.drop_duplicates(inplace=True)
    overall = overall.sort_values("Ditto_Deleterious", ascending=False)

    overall.to_csv('./predictions/ditto/'+args.input, index=False, compression="gzip")

    overall.head(100).to_csv('./predictions/ditto/100_'+args.input, index=False, compression="gzip")
    del overall
    return None

def read_vcf(path):
    with gzip.open(path, 'rt') as f:
        lines = [l for l in f if not l.startswith('##')]
    return pd.read_csv(
        io.StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'REF': str, 'ALT': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input sample name for predictions",
    )

    args = parser.parse_args()
    main(args)

