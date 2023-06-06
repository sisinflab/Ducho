# Ducho

This is the official GitHub repo for the paper "_Ducho: A Unified Framework for the Extraction of Multimodal
Features in Recommendation_", under review at ACM Multimedia 2023 in the Open Source track.

## Table of contents

- [What is Ducho](#what-is-ducho)
- [How can I run Ducho](#how-can-i-run-ducho)
- [Prerequisites](#prerequisites)
  - [Local](#local)
  - [Docker](#docker)
  - [Google Colab](#google-colab)

## What is Ducho

Ducho is a Python framework for the extraction of multimodal features for recommendation. It provides a unified interface to most of the common libraries for deep learning (e.g., TensorFlow, PyTorch, Transformers) to extract high-level features from items (e.g., product images/descriptions) and user-item interactions (e.g., users reviews). It is highly configurable through a YAML-based configuration file (which may be override by input arguments from the command line in case). Users can indicate the source from which to extract the multimodal features (i.e., items/interactions), the modalities (i.e., visual/textual/audio), and the list of models along with output layers and preprocessing steps to extract the features.

## How can I run Ducho

You may choose among three options:

- Locally by cloning this GitHub repo.
- By pulling our docker image on DockerHub ([link](https://hub.docker.com/repository/docker/sisinflabpoliba/ducho/general)).
- On Google Colab ([link](https://colab.research.google.com/drive/1ouKkdxOObOL0BI00r0c157oNRqwxqTgt)).

## Prerequisites

### Local

Ducho may work on both CPU and GPU. However, if you want to speed up your feature extraction, we highly recommend to go for the GPU-accelerated option. 

In that case, if you machine is equipped with NVIDIA drivers, you should first make sure CUDA is installed along with the compatible NVIDIA drivers. 

For example, a possible working environment involves the following (you may refer to any Google Colab notebook):

```
Nvidia drivers: 525.85.12
Cuda: 11.8.89
Python: 3.10.11
Pip: 23.1.2
```

Please, refer to this [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for the official guidelines on how to install the nvidia toolkit on linux from scratch.

### Docker

Docker is easily the best option to run any NVIDIA/CUDA-based framework, because it provides docker images with everything already setup for almost all your needs.

First of all, you need to install the docker engine on your machine. Here is the official [link](https://docs.docker.com/engine/install/ubuntu/) for ubuntu. Then, for the sake of the demos provided in this repository, you might also need docker compose (here is a reference [link](https://docs.docker.com/compose/install/standalone/)).

Quite conveniently, you can find several CUDA-equipped images on Docker Hub. You may refer to this [link](https://hub.docker.com/r/nvidia/cuda ). Depending on your CUDA version, you may also need to install nvidia-docker2 (in case, here is a [reference](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

To test if everything worked smoothly, pull and run a container from the following docker image through this command:

```
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi
```

Once the docker image has been downloaded from the hub, you should be able to see something like this:

```
==========
== CUDA ==
==========

CUDA Version 11.8.0

Container image Copyright (c) 2016-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

Tue Jun  6 14:11:59 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.129.06   Driver Version: 470.129.06   CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000001:00:00.0 Off |                    0 |
| N/A   33C    P0    26W /  70W |      0MiB / 15109MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
meaning that the installation is ok and you are finally ready to pull ducho's image (which is actually built from this CUDA image) from Docker Hub!

### Google Colab

You just need a Google Drive account!
