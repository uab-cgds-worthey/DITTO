#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Define the command-line options to specify the path to VCF files
params.sample_sheet = '.test_data/file_list_partaa'
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
         sample_sheet     : ${params.sample_sheet}
         output_dir       : ${output_dir}
         oc_modules       : ${params.oc_modules}
         """
         .stripIndent()


// Define the process to run 'oc' with the specified parameters
process runOC {

  // Define the conda environment file to be used
  conda '../configs/envs/open-cravat.yaml'

  input:
  path var_ch
  val var_build
  val oc_mod_path

  output:
  path "${var_ch}.variant.csv"

  script:
  """
  oc config md ${oc_mod_path}
  oc run ${var_ch} -l ${var_build} -t csv --mp 2 --package mypackage -d .
  rm -rf ${var_ch}.sqlite ${var_ch}.err
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

  script:
  """
  python ${baseDir}/src/annotation_parsing/parse.py -i ${var_ann_ch} -e parse -o . -c ${baseDir}/configs/opencravat_test_config.json
  """
}

// Define the process for prediction
process prediction {

  // Define the conda environment file to be used
  conda '../configs/envs/ditto-nf.yaml'

  input:
  path var_parse_ch

  script:
  """
  python ${baseDir}/src/predict/predict.py -i ${var_parse_ch} -o ${output_dir} -c ${baseDir}/configs/col_config.yaml -d ${baseDir}/model/
  """
}

// Define the workflow by connecting the processes
// 'vcfFile' will be the channel containing the input VCF files
// Each file in the channel will be processed through the steps defined above.
workflow {

  // Define input channels for the VCF files
  vcfFile = Channel.fromPath(params.sample_sheet).splitCsv(header: false)
  vcfBuild = params.build
  oc_mod_path = params.oc_modules

  // Run processes
  runOC(vcfFile,vcfBuild,oc_mod_path )
  parseAnnotation(runOC.out)
  // Scatter the output of parseAnnotation to process each file separately
  parseAnnotation.out.flatten().set { parsed_files }
  prediction(parsed_files)
}
