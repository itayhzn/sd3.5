#! /bin/sh

#SBATCH --job-name=plots
#SBATCH --output=/wolflab/itaytuviah/sd3.5/jobs-out-err/%x-%j.out
#SBATCH --error=/wolflab/itaytuviah/sd3.5/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=1500 # max time (minutes)
#SBATCH --gres=gpu:A100:1
#SBATCH --account=gpu-tad-users_v2
#SBATCH --partition=gpu-tad-pool
#SBATCH --mem-per-gpu=55000
#SBATCH --qos=owner

###### ignore SBATCH --mem-per-gpu=55000
###### salloc -A gpu-tad-users_v2 -p gpu-tad-pool --qos=owner --gres=gpu:A100:1 --time=02:00:00 --job-name=vscode && srun --pty bash

srun --gres=gpu:1 nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export USER_DIR="/wolflab/itaytuviah"
export PROJECT_DIR="${USER_DIR}/sd3.5"
export PATH="${USER_DIR}/miniconda3/bin:${PATH}"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# keep all caches/pkgs off $HOME
export XDG_CACHE_HOME=${USER_DIR}/.cache
export PIP_CACHE_DIR=${USER_DIR}/.cache/pip
export TMPDIR=${USER_DIR}/tmp
export CONDA_PKGS_DIRS=${USER_DIR}/.cache/conda/pkgs
export CONDA_ENVS_PATH=${USER_DIR}/conda-envs

conda run -n myenv python reinforce_main.py --lr 1e-4 --schedule 4 --num_epochs 30 --state_alpha 0.05 --action_alpha 1 --experiment_name T4_LR1e-4_Ioi_Pdance_S23 --group_size 8 --out_dir "outputs/grpo-nn/dancer" --prompts "A woman performing an intricate dance on stage..." --seeds 23 --faiss_index "faiss_indexes/open-images-03"
