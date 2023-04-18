from pathlib import Path


WORKFLOW_PATH = Path(workflow.basedir).parent

configfile:  WORKFLOW_PATH / "configs/some_workflow_config.yaml"
SAMPLE_LIST = config["samples"]


RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")



wildcard_constraints:
    sample="|".join(SAMPLE_LIST) #"TRAIN_12|TRAIN_13"


rule all:
    input:
        # define target files here


rule some_name:
    input:
        # input file
    output:
        # output file
        PROCESSED_DIR / "some_outfile.txt"
    message:
        "some message"
    conda:
        str(WORKFLOW_PATH / "configs/envs/bcftools.yaml")
    shell:
        r"""
        sopme command
        """
