#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define the command-line options to specify the path to VCF files
params.vcf_path = '.test_data/testing_variants_hg38.vcf.gz'
params.hg38 = "/data/project/worthey_lab/datasets_central/human_reference_genome/processed/GRCh38/no_alt_rel20190408/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"

// Define the Scratch directory
def scratch_dir = System.getenv("USER_SCRATCH") ?: "/tmp"

params.outdir = "${scratch_dir}"



// Define the output directory for intermediate and final results
output_dir = params.outdir

log.info """\
         D I T T O - N F   P I P E L I N E
         ===================================
         hg38             : ${params.hg38}
         into_vcf         : ${params.vcf_path}
         output_dir       : ${output_dir}
         """
         .stripIndent()


// Define the process to extract the required information from VCF and convert to txt.gz
process extractFromVCF {
  publishDir output_dir, mode:'copy'

  // Define the input channel for the VCF files
  input:
  path into_vcf

  output:
    path("${into_vcf.baseName}.txt.gz")

  // Modify the path if necessary.
  shell:
  """
  zcat ${into_vcf} | grep -v "^#" | cut -d\$'\t' -f1,2,4,5 | grep -v "*" | head -5 | gzip > ${into_vcf.baseName}.txt.gz
  """
}

// Define the process to run 'oc' with the specified parameters
process runOC {
  publishDir output_dir, mode:'copy'

  input:
  path var_ch

  output:
    path("${var_ch}.variant.csv")

  shell:
  """
  oc run ${var_ch} -l hg38 -t csv --package mypackage -d ${output_dir}
  cp ${output_dir}/${var_ch}.variant.csv .
  """
}

// Define the process to parse the annotation
process parseAnnotation {
  publishDir output_dir, mode:'copy'

  input:
  path var_ann_ch

  output:
    path("${var_ann_ch.baseName}.csv.gz")

  script:
  """
  python ${baseDir}/src/annotation_parsing/parse.py -i ${var_ann_ch} -e parse -o ${var_ann_ch.baseName}.csv.gz -c ${baseDir}/configs/opencravat_test_config.json
  """
}

// Define the process for prediction
process prediction {
  publishDir output_dir, mode:'copy'

  input:
  path var_parse_ch

  script:
  """
  python ${baseDir}/src/predict/predict.py -i ${var_parse_ch} -o ${output_dir} -c ${baseDir}/configs/col_config.yaml -d ${baseDir}/data/processed/train_data_3_star/
  """
}

// Define the workflow by connecting the processes
// 'into_vcf' will be the channel containing the input VCF files
// Each file in the channel will be processed through the steps defined above.
workflow {
  extractFromVCF(channel.fromPath(params.vcf_path))
  runOC(extractFromVCF.out)
  parseAnnotation(runOC.out)
  prediction(parseAnnotation.out)
}