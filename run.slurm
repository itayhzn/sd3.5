#! /bin/sh

#SBATCH --job-name=conda_install
#SBATCH --output=/wolflab/itaytuviah/sd3.5/jobs-out-err/%x-%j.out
#SBATCH --error=/wolflab/itaytuviah/sd3.5/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=1500 # max time (minutes)
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=gpu-tad-pool

###### ignore SBATCH --mem-per-gpu=55000
###### ignore SBATCH --gpus=0
###### ignore SBATCH --constraint=h100

# srun --gres=gpu:1 nvidia-smi
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

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

# export PYTORCH_KERNEL_CACHE_PATH="${PROJECT_DIR}/cache"
# export TORCH_KERNEL_CACHE_DIR="${PROJECT_DIR}/cache"
# export HF_HOME="${PROJECT_DIR}/cache"
# export TRANSFORMERS_CACHE="${PROJECT_DIR}/cache"
# export WANDB_CACHE_DIR="${PROJECT_DIR}/cache"
# export WANDB_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_CONFIG_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_TEMP_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_DATA_DIR="${PROJECT_DIR}/cache/wandb/"

# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame."   \
#                     --seeds 1024 

# prove CUDA works & show torch version
conda run -n myenv python -c "import torch; print(torch.__version__); import flash_attn; print(\"flash-attn OK\")"
nvidia-smi