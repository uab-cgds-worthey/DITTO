################### 2/21/21 ######################################################################
Current workflow:
`module load Anaconda3/2020.02
module load tabix
module load BCFtools`
Download Clinvar - 
    `wget -P /data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/data/external/ https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
    tabix -fp vcf clinvar.vcf.gz `
Check chromosomes and keep note for modifications - 
    `zgrep -v ^# clinvar.vcf.gz | cut -f1 -d$'\t' | sort -u`
Copy HGMD VCF - 
    `cp /data/project/worthey_lab/manual_datasets_central/hgmd/2020q4/hgmd_pro_2020.4_hg38.vcf ./`
Check chromosomes and keep note for modifications - 
    `grep -v ^# hgmd_pro_2020.4_hg38.vcf | cut -f1 -d$'\t' | sort -u`
Fix the INFO column and index for merging - 
    `sed -E 's/(^[^#]+)(=")([^;"]+)(;)+([^;]*?)(")/\1\2\3%3B\5\6/' hgmd_pro_2020.4_hg38.vcf > hgmd_pro_2020.4_hg38_fixed_info.vcf`
    bgzip -c  hgmd_pro_2020.4_hg38_fixed_info.vcf > hgmd_pro_2020.4_hg38_fixed_info.vcf.gz
    tabix -fp vcf hgmd_pro_2020.4_hg38_fixed_info.vcf.gz `
Merge Clinvar and HGMD - 
    `bcftools merge clinvar.vcf.gz hgmd_pro_2020.4_hg38_fixed_info.vcf.gz -Ov -o ../interim/merged.vcf`
Add `chr` to chromosomes columns - 
    `sed -E -i 's/(^[^#]+)/chr\1/' ../interim/merged.vcf `
Fix chromosome issues noted before - 
    `sed -i 's/^chrMT/chrM/g' ../interim/merged.vcf 
    grep -v ^chrNW ../interim/merged.vcf > ../interim/merged_chr_fix.vcf`
Check chromosomes and fix any remaining issues - 
    `grep -v ^# ../interim/merged_chr_fix.vcf | cut -f1 -d$'\t' | sort -u`
Normalize the variants using reference genome - 
    `bcftools norm -f /data/project/worthey_lab/datasets_central/human_reference_genome/processed/GRCh38/no_alt_rel20190408/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna ../interim/merged_chr_fix.vcf -Oz -o ../interim/merged_norm.vcf.gz`
Filter variants by size (<30kb) and class - 
    `python ../../src/training/data-prep/extract_variants.py`
    Clinvar variants: 300280
    HGMD variants: 156402
bgzip and Tabix index the file - 
    `bgzip -c ../interim/merged_sig_norm.vcf > ../interim/merged_sig_norm.vcf.gz
    tabix -fp vcf ../interim/merged_sig_norm.vcf.gz`
Parse dbNSFP database to work with VEP (needed to format transcripts to separate lines) - 
    `cd variant_annotation/formatting/`
    ```
        python dbnsfp_vep_formatter.py -i /data/project/worthey_lab/temp_datasets_central/mana/dbnsfp/processed/v4.1a_20200616/dbNSFP4.1a_variant.complete.bgz -o /data/project/worthey_lab/projects/       experimental_pipelines/tarun/ditto/variant_annotation/formatting/dbNSFP4.1a_variant.complete.tsv
        bgzip -c dbNSFP4.1a_variant.complete.tsv > dbNSFP4.1a_variant.complete.gz
        tabix -f dbNSFP4.1a_variant.complete.gz
    ```
Copy paths to dataset yaml file - 
    ```
    cadd_snv: "/data/project/worthey_lab/temp_datasets_central/mana/cadd/raw/hg38/v1.6/whole_genome_SNVs.tsv.gz"
    cadd_indel: "/data/project/worthey_lab/temp_datasets_central/mana/cadd/raw/hg38/v1.6/gnomad.genomes.r3.0.indel.tsv.gz"
    gerp: "/data/project/worthey_lab/temp_datasets_central/mana/gerp/processed/hg38/v1.6/gerp_score_hg38.bg.gz"
    gnomad_genomes: "/data/project/worthey_lab/temp_datasets_central/mana/gnomad/v3.0/data/gnomad.genomes.r3.0.sites.vcf.bgz"
    clinvar: "/data/project/worthey_lab/temp_datasets_central/mana/clinvar/data/grch38/20210119/clinvar_20210119.vcf.gz"
    dbNSFP: "/data/project/worthey_lab/projects/experimental_pipelines/tarun/ditto/variant_annotation/formatting/dbNSFP4.1a_variant.complete.gz"
    ```
Run variant annotation as shown in ReadMe file - 
    `./src/run_pipeline.sh -s -v ../data/interim/merged_sig_norm.vcf.gz -o ../data/interim -d ~/.ditto_datasets.yaml`
Parse the annotated vcf file - 
    `python parse_annotated_vars.py -i ../data/processed/merged_sig_norm_vep-annotated.vcf.gz -o ../data/processed/merged_sig_norm_vep-annotated.tsv`
Extract Class information for all these variants - 
    `python extract_class.py`
Filter, stats and prep the data - 
    `python filter.py`