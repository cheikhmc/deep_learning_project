#!/bin/bash
#SBATCH -c 12
#SBATCH -o job.%j
#SBATCH -p node_name
source yourenvpath
time python3 deep_learning_project/models/transfer_learning.py
