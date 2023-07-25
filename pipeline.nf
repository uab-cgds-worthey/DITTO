#!/usr/bin/env nextflow

// Define the command-line options to specify the path to VCF files
params.vcf_path = 'path/to/vcf/files/*PROBAND.vcf.gz'
params.hg38 = "${baseDir}/data/hg38.fasta"
params.outdir = "${baseDir}/data/processed/"

// Define the input channel for the VCF files
input:
  file(params.vcf_path) into_vcf

// Define the output directory for intermediate and final results
output_dir = params.outdir

log.info """\
         D I T T O - N F   P I P E L I N E
         ===================================
         hg38         : ${params.hg38}
         input        : ${params.vcf_path}
         outdir       : ${params.outdir}
         """
         .stripIndent()


// Define the process to extract the required information from VCF and convert to txt.gz
process extractFromVCF {

  // Modify the path if necessary.
  script:
  """
  source activate opencravat
  zcat ${into_vcf} | grep -v "^#" | cut -d\$'\t' -f1,2,4,5 | grep -v "*" | gzip > ${output_dir}/${into_vcf.baseName}.txt.gz
  """
}

// Define the process to run 'oc' with the specified parameters
process runOC {
  // Use the 'source' directive to activate the 'training' environment
  // Modify the path if necessary.
  script:
  """
  source activate training
  oc run ${output_dir}/${into_vcf.baseName}.txt.gz -l hg38 -t csv --package mypackage -d ${output_dir}
  """
}

// Define the process to parse the annotation
process parseAnnotation {
  // Use the 'source' directive to activate the 'training' environment
  // Modify the path if necessary.
  script:
  """
  source activate training
  python ${baseDir}/src/annotation_parsing/parse.py -i ${output_dir}/${into_vcf.baseName}.txt.gz.variant.csv.gz -e parse -o ${output_dir}/${into_vcf.baseName}.csv.gz -c ${baseDir}/configs/opencravat_test_config.json
  """
}

// Define the process for prediction
process prediction {
  // Use the 'source' directive to activate the 'training' environment
  // Modify the path if necessary.
  script:
  """
  source activate training
  python ${baseDir}/src/predict/predict.py -i ${output_dir}/${into_vcf.baseName}.csv.gz -o ${output_dir} -c ${baseDir}/configs/col_config.yaml -d ${baseDir}/data/processed/train_data_3_star/
  """
}

// Define the workflow by connecting the processes
// 'into_vcf' will be the channel containing the input VCF files
// Each file in the channel will be processed through the steps defined above.
workflow {
  extractFromVCF()
  runOC
  parseAnnotation()
  prediction()
}
