FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y unzip git curl software-properties-common wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb && \
    dpkg -i libcudnn8_8.6.0.163-1+cuda11.8_amd64.deb && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/sisinflab/Ducho.git && \
    apt-get install -y python3.8-dev && \
    pip install --upgrade pip && \
    pip install -r Ducho/requirements.txt --ignore-installed

WORKDIR Ducho