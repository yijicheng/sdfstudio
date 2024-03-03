FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Basic setup
ENV CMAKE_VERSION=3.22.2
ENV PYTHON_VERSION=3.8

ENV PATH="/usr/bin/cmake/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        wget vim pkg-config unzip rsync \
        ninja-build x11-apps \
    && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
        -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /usr/bin/cmake \
    && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
    && rm /tmp/cmake-install.sh \
    && apt-get install sudo \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Set working directory
WORKDIR /opt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install Miniconda with given python version
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && chmod +x ~/miniconda.sh \
     && ~/miniconda.sh -b -p /opt/conda \
     && rm ~/miniconda.sh \
     && /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
	 && /opt/conda/bin/conda clean -ya
ENV PATH=/opt/conda/bin:$PATH
ENV PATH=/root/.local/bin:$PATH 

# Install required libraries
RUN conda install -y pytorch=1.12.1 torchvision cudatoolkit=11.3 -c pytorch
    # && conda install -y tensorboard matplotlib scikit-image scipy jupyter  \
    #     ninja cython typing future pytest black isort flake8 scikit-learn \
    # && /opt/conda/bin/python -m pip install -U wandb python-dotenv pre-commit nbstripout \
    #     hydra-core hydra-colorlog hydra-optuna-sweeper rich pytorch-lightning torchmetrics


ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV PATH=$PATH:$CUDA_HOME/bin

# RUN git clone --recurse-submodules -j8 https://github.com/NVlabs/tiny-cuda-nn.git \
#         && cd /opt/tiny-cuda-nn/bindings/torch \
#         && pip install -e .

WORKDIR /