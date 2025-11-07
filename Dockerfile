
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

SHELL [ "bash", "-c" ]

#-------------------------------------------------
# 1) System packages
#-------------------------------------------------
RUN apt update && \
    apt install -yq \
        ffmpeg \
        build-essential \
        curl \
        wget \
        git

#-------------------------------------------------
# 2) User setup
#-------------------------------------------------
# USER vscode

# RUN ln -s /usr/local/cuda-12.6 /usr/local/cuda || true

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Git LFS
# RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
#     sudo apt-get install -yq git-lfs && \
#     git lfs install

#-------------------------------------------------
# 3) Miniconda installation
#-------------------------------------------------
RUN cd /tmp && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /root/miniconda3 && \
    rm ./Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/root/miniconda3/bin:${PATH}"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true

#-------------------------------------------------
# 4) Copy environment file and create environment
#-------------------------------------------------
WORKDIR /storage/itaytuviah/sd3.5

COPY environment.yml /tmp/environment.yml

RUN conda env remove -n myenv -y || true && \
    conda env create -f /tmp/environment.yml -y && \
    conda clean -afy

RUN conda run -n myenv bash -lc 'which nvcc && nvcc --version'

RUN conda run -n myenv python -c "import torch,os,sys; \
print('python:', sys.version.split()[0]); \
print('torch:', torch.__version__); \
print('torch.version.cuda:', torch.version.cuda); \
print('is_available:', torch.cuda.is_available()); \
print('CUDA_HOME:', os.environ.get('CUDA_HOME'))"

#RUN conda run -n llava_yaml pip install flash-attn --no-build-isolation --no-cache-dir
RUN if command -v nvcc >/dev/null 2>&1; then \
        echo "✅ CUDA detected — installing flash-attn..." && \
        CUDA_HOME=/usr/local/cuda \
        # conda run -n myenv pip install flash-attn --no-build-isolation --no-cache-dir; \
        conda run -n myenv python -m pip install --no-cache-dir \
        https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl; \
        # --no-cache-dir --find-links https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/ flash-attn==2.8.3; \
    else \
        echo "⚠️  Skipping flash-attn install — CUDA not found"; \
    fi

#-------------------------------------------------
# 5) ImageReward installation
#-------------------------------------------------
RUN conda env remove -n rewardenv -y || true && \
    conda create -n rewardenv python=3.8 -y && \
    conda run -n rewardenv pip install clip image-reward && \
    conda clean -afy

#-------------------------------------------------
# 6) Create FAISS index directory
#-------------------------------------------------
WORKDIR /storage/itaytuviah/sd3.5

COPY download_dataset.py /tmp/download_dataset.py
RUN conda run -n myenv python /tmp/download_dataset.py && \
    conda run -n myenv python -c "from build_index import build_faiss_indexes; build_faiss_indexes(dataset_dirs=['datasets/ksaml/Stanford_dogs'], out_dir='dog_index')"
    
RUN rm -rf 'datasets/ksaml/Stanford_dogs'

#-------------------------------------------------
# 6) Entry point
#-------------------------------------------------
# Instead of "source activate", which can be tricky in non-interactive shells,
# use 'conda run' to ensure the environment is active when your script runs.
# run python main.py
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "main.py"]