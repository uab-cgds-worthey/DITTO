# DITTO

Diagnosis prediction tool using AI.


Install packages and dependencies
-------------------------------------

srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=200G --time=01:59:00 --partition=pascalnodes --job-name=classify --pty /bin/bash

Create an environment to work on - 

    `module load Anaconda3/2020.02`

For first time users only - 

    `conda env create --name training --file configs/environment.yaml`

Activate your environment-

    `source activate training`

