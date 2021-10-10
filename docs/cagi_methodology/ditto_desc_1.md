# Methodology

VCF files provided by CAGI-RGP were retrieved and preprocessing performed including splitting of joint VCF to single
sample VCF, variant normalization, splitting of multi-allelic sites, removal of poor quality variants based on depth and
genotype quality, and variant annotation with VEP. Our machine learning model (Ditto) was trained using pathogenic and
benign small variants retrieved from ClinVar and HGMD, to predict variant pathogenicity classifications. Ranking of
these classification predictions combined with phenotype-based ranking of genes was used to derive an overall ranking of
the top-100 candidate causal variants for each proband.
