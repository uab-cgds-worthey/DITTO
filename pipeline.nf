#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define the command-line options to specify the path to VCF files
params.vcf_path = '.test_data/*'
params.build = "hg38"
params.oc_modules = "/data/project/worthey_lab/projects/experimental_pipelines/tarun/opencravat/modules"
// Define the Scratch directory
def scratch_dir = System.getenv("USER_SCRATCH") ?: "/tmp"

params.outdir = "${scratch_dir}"

// Define the output directory for intermediate and final results
output_dir = params.outdir

log.info """\

         D I T T O - N F   P I P E L I N E
         ===================================
         Parameters:
         build            : ${params.build}
         vcf_file         : ${params.vcf_path}
         output_dir       : ${output_dir}
         oc_modules       : ${params.oc_modules}
         """
         .stripIndent()

// Define the process to extract the required information from VCF and convert to txt.gz
process extractFromVCF {
  publishDir output_dir, mode:'copy'

  // Define the conda environment file to be used
  // conda 'configs/envs/bcftools.yaml'

  // Define the input channel for the VCF files
  input:
  path homref_vcf

  output:
  path "${homref_vcf.baseName}.txt.gz"//, emit: extractedVCF

  // Specify memory and partition requirements for the process
  memory = '5G'
  cpus = 1
  time = '1h'

  shell:
  """
  zcat ${homref_vcf} | grep -v "^#" | cut -d\$'\t' -f1,2,4,5 | grep -v "*" | gzip > ${homref_vcf.baseName}.txt.gz
  """
}

// Define the process to run 'oc' with the specified parameters
process runOC {

  // Define the conda environment file to be used
  conda 'configs/envs/open-cravat.yaml'

  input:
  path var_ch
  val var_build
  val oc_mod_path

  output:
  path "*.variant.csv"

  // Specify memory and partition requirements for the process
  memory = '100G'
  cpus = 20
  time = '50h'

  script:
  """
  oc config md ${oc_mod_path}
  oc module install-base
  oc run ${var_ch} -l ${var_build} -t csv --package mypackage -d .
  """

}

// Define the process to parse the annotation
process parseAnnotation {

  // Define the conda environment file to be used
  conda 'python=3.10'

  input:
  path var_ann_ch

  output:
  path "*_parsed.csv.gz"

  // Specify memory and partition requirements for the process
  memory = '10G'
  cpus = 5
  time = '2h'

  script:
  """
  python ${baseDir}/src/annotation_parsing/parse.py -i ${var_ann_ch} -e parse -o . -c ${baseDir}/configs/opencravat_test_config.json
  """
}

// Define the process for prediction
process prediction {

  // Define the conda environment file to be used
  conda 'configs/envs/ditto-nf.yaml'

  input:
  path var_parse_ch

  // Specify memory and partition requirements for the process
  memory = '20G'
  cpus = 5
  time = '2h'

  script:
  """
  python ${baseDir}/src/predict/predict.py -i ${var_parse_ch} -o ${output_dir} -c ${baseDir}/configs/col_config.yaml -d ${baseDir}/data/processed/train_data_3_star/
  """
}

// Define the workflow by connecting the processes
// 'vcfFile' will be the channel containing the input VCF files
// Each file in the channel will be processed through the steps defined above.
workflow {

  // Define input channels for the VCF files
  vcfFile = Channel.fromPath(params.vcf_path)
  vcfBuild = params.build
  oc_mod_path = params.oc_modules

  // Run processes
  extractFromVCF(vcfFile)
  runOC(extractFromVCF.out,vcfBuild,oc_mod_path )
  parseAnnotation(runOC.out)
  // Scatter the output of parseAnnotation to process each file separately
  parseAnnotation.out.flatten().set { files }
  prediction(files)
}
