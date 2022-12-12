#!/bin/bash
#SBATCH -c 12
#SBATCH -o job.%j
#SBATCH -p node_name
source your_env_path
time python3 CNN_.py
