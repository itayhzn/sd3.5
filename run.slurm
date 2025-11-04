#! /bin/sh

#SBATCH --job-name=conda_install
#SBATCH --output=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.out
#SBATCH --error=/home/ai_center/ai_users/itaytuviah/video-motion/jobs-out-err/%x-%j.err
#SBATCH --open-mode=append      # Append instead of overwriting logs
#SBATCH --time=1500 # max time (minutes)
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-tad-pool

###### ignore SBATCH --mem-per-gpu=55000
###### ignore SBATCH --gpus=0
###### ignore SBATCH --constraint=h100

# srun --gres=gpu:1 nvidia-smi
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export USER_DIR="/wolflab/itaytuviah"
export PROJECT_DIR="${USER_DIR}/sd3.5"

# export PYTORCH_KERNEL_CACHE_PATH="${PROJECT_DIR}/cache"
# export TORCH_KERNEL_CACHE_DIR="${PROJECT_DIR}/cache"
# export HF_HOME="${PROJECT_DIR}/cache"
# export TRANSFORMERS_CACHE="${PROJECT_DIR}/cache"
# export WANDB_CACHE_DIR="${PROJECT_DIR}/cache"
# export WANDB_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_CONFIG_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_TEMP_DIR="${PROJECT_DIR}/cache/wandb/"
# export WANDB_DATA_DIR="${PROJECT_DIR}/cache/wandb/"

export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${USER_DIR}/miniconda3

export PATH="${USER_DIR}/miniconda3/bin:${PATH}"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

conda env remove -n myenv -y || true
conda env create -f ${PROJECT_DIR}/environment.yml -y 
conda clean -afy

if command -v nvcc >/dev/null 2>&1; then \
    echo "✅ CUDA detected — installing flash-attn..." && \
    CUDA_HOME=/usr/local/cuda \
    # conda run -n myenv pip install flash-attn --no-build-isolation --no-cache-dir; \
    conda run -n myenv python -m pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl; \
    # --no-cache-dir --find-links https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/ flash-attn==2.8.3; \
else \
    echo "⚠️  Skipping flash-attn install — CUDA not found"; \
fi

conda env remove -n rewardenv -y || true
conda create -n rewardenv python=3.8 -y 
conda run -n rewardenv pip install clip image-reward
conda clean -afy


# python generate.py --task t2v-1.3B \
#                     --size 832*480 \
#                     --ckpt_dir ./Wan2.1-T2V-1.3B \
#                     --prompts "A woman performing an intricate dance on stage, illuminated by a single spotlight in the first frame."   \
#                     --seeds 1024 