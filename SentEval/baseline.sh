#!/bin/bash

#####################
# job-array example #
#####################

#SBATCH --job-name=phi3

# 16 jobs will run in this array at the same time
#SBATCH --array=1-1

# run for five minutes
#              d-hh:mm:ss
#SBATCH --time=1-00:00:00

# default qos should do it



# 500MB memory per core   
# this is a hard limit 
#SBATCH --mem-per-cpu=30000MB



#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-medium
#SBATCH --account=cml-zhou

#SBATCH --gres=gpu:a100:1

#SBATCH --partition=tron
#SBATCH --qos=high
#SBATCH --account=nexus

#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger

#SBATCH --gres=gpu:rtxa6000:1

# you may not place bash commands before the last SBATCH directive

# . /usr/share/Modules/init/bash
# . /etc/profile.d/ummodules.sh
module load Python/3.7.6
module load cuda/11.7.0

source ../../LLM-Adapters/env38/bin/activate

echo "now processing task id:: " ${SLURM_ARRAY_TASK_ID}

nvidia-smi

# wandb online
# wandb disabled
# export TORCH_HOME="/fs/nexus-scratch/litzy/models/"

TOKENIZERS_PARALLELISM=false

# export XDG_CACHE_HOME="/fs/nexus-scratch/litzy/"
export XDG_CACHE_HOME="/fs/cml-scratch/litzy/"


# cd /nfshomes/litzy/UMD2024/SentEval/data/downstream

# chmod +x get_transfer_data.bash

# ./get_transfer_data.bash

python -u bert-avg.py


# happy end
exit 0