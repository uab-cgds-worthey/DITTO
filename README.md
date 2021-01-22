# DITTO

Diagnosis prediction tool using AI.

Install packages and dependencies
-------------------------------------
srun --ntasks=1 --cpus-per-task=1 --mem-per-cpu=200G --time=01:59:00 --partition=pascalnodes --job-name=classify --pty /bin/bash
Using conda envi to install and work on tools
    module load Anaconda3/2020.02
    conda env create --name training --file config/environment.yaml 


