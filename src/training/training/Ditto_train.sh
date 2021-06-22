#!/bin/bash
for i in {0..4} 
do 
    python slurm-launch.py --exp-name Ditto_tuning --command "python optuna-tpe-stacking_training.ipy --vtype snv_protein_coding" 
    sleep 60 
    rm Ditto_tuning*
done