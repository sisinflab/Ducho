FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y unzip git curl software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3.8-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    git clone https://github.com/sisinflab/Ducho.git && \
    apt-get install -y python3.8-dev && \
    pip install --upgrade pip && \
    pip install -r Ducho/requirements_docker.txt --ignore-installed

WORKDIR Ducho