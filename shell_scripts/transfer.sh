#!/bin/bash
#SBATCH -c 12
#SBATCH -o job.%j
#SBATCH -p node_name
source yourenvpath
time python3 transfer_learning.py
