# slurm-launch.py
# Usage:
# for i in {0..4}; do python slurm-launch.py --exp-name Ditto_tuning --command "python optuna-tpe-stacking_training.ipy --vtype snv_protein_coding" sleep 60 ; done

import argparse
import subprocess
import sys
import time

template_file = "slurm-template.sh" #Path(__file__) / 
JOB_NAME = "${JOB_NAME}"
NUM_NODES = "${NUM_NODES}"
NUM_GPUS_PER_NODE = "${NUM_GPUS_PER_NODE}"
NUM_CPUS_PER_NODE = "${NUM_CPUS_PER_NODE}"
TOT_MEM = "${TOT_MEM}"
PARTITION_OPTION = "${PARTITION_OPTION}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).")
    parser.add_argument(
        "--num-nodes",
        "-n",
        type=int,
        default=1,
        help="Number of nodes to use.")
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the "
        "return of 'sinfo'. Default: ''.")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="Number of GPUs to use in each node. (Default: 0)")
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=2,
        help="Number of CPUs to use in each node. (Default: 2)")
    parser.add_argument(
        "--mem",
        type=str,
        default="15G",
        help="Total Memory to use. (Default: 5G)")
    parser.add_argument(
        "--partition",
        type=str,
        default="short",
        help="Default partition: short"
    )
    parser.add_argument(
        "--load-env",
        type=str,
        default="training",
        help="The script to load your environment ('training')")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python optuna-tpe-stacking_results.ipy'. "
        "Note that the command must be a string.")
    args = parser.parse_args()

    if args.node:
        # assert args.num_nodes == 1
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name,
                              time.strftime("%m%d-%H%M%S", time.localtime()))

    partition_option = "#SBATCH --partition={}".format(
        args.partition) if args.partition else ""

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = text.replace(JOB_NAME, job_name)
    text = text.replace(NUM_NODES, str(args.num_nodes))
    text = text.replace(NUM_GPUS_PER_NODE, str(args.num_gpus))
    text = text.replace(NUM_CPUS_PER_NODE, str(args.num_cpus))
    text = text.replace(TOT_MEM, str(args.mem))
    text = text.replace(PARTITION_OPTION, str(args.partition))
    text = text.replace(COMMAND_PLACEHOLDER, str(args.command))
    text = text.replace(LOAD_ENV, str(args.load_env))
    text = text.replace(GIVEN_NODE, node_info)
    text = text.replace(
        "# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO "
        "PRODUCTION!",
        "# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE "
        "RUNNABLE!")

    # ===== Save the script =====
    script_file = "{}.sh".format(job_name)
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file])
    print(
        "Job submitted! Script file is at: <{}>. Log file is at: <{}>".format(
            script_file, "{}.log".format(job_name)))
    sys.exit(0)