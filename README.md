# Ducho

<img src="https://raw.githubusercontent.com/sisinflab/Ducho/main/ducho.png"  width="600">

This is the official GitHub repo for the paper "_Ducho: A Unified Framework for the Extraction of Multimodal
Features in Recommendation_", under review at ACM Multimedia 2023 in the Open Source track.

## Table of contents

- [What is Ducho](#what-is-ducho)
- [How can I run Ducho](#how-can-i-run-ducho)
- [Prerequisites](#prerequisites)
  - [Local](#local)
  - [Docker](#docker)
  - [Google Colab](#google-colab)
- [Installation](#installation)
- [Try Ducho](#try-ducho)
- [Use Ducho](#use-ducho)

## What is Ducho

<table style="border-collapse:collapse;border-spacing:0" class="tg"><thead><tr><th style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:normal;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"></th><th style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal" colspan="2">Sources</th><th style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal" colspan="3">Backends</th></tr></thead><tbody><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;font-weight:bold;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Modalities</td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Items</td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Interactions</td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.gstatic.com/devrel-devsite/prod/v37463d4834445c1e880de1e91d2f8fc2c6a0e86fca4aa6a7bdbb270b040181dc/tensorflow/images/lockup.svg" width="90" height="20"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="Image" width="62" height="15"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png" width="99" height="16"></td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Audio</td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><a href="https://pytorch.org/audio/stable/index.html" target="_blank" rel="noopener noreferrer">Torchaudio</a></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Visual</td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><a href="https://pytorch.org/vision/stable/index.html" target="_blank" rel="noopener noreferrer">Torchvision</a></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"></td></tr><tr><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal">Textual</td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><img src="https://www.freepnglogos.com/uploads/tick-png/check-mark-tick-vector-graphic-21.png" alt="Image" width="25" height="24"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"></td><td style="border-color:inherit;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;overflow:hidden;padding:10px 5px;text-align:center;vertical-align:top;word-break:normal"><a href="https://www.sbert.net/" target="_blank" rel="noopener noreferrer">SentenceTransformers</a></td></tr></tbody></table>

Ducho is a Python framework for the extraction of multimodal features for recommendation. It provides a unified interface to most of the common libraries for deep learning (e.g., TensorFlow, PyTorch, Transformers) to extract high-level features from items (e.g., product images/descriptions) and user-item interactions (e.g., users reviews). It is highly configurable through a YAML-based configuration file (which may be override by input arguments from the command line in case). Users can indicate the source from which to extract the multimodal features (i.e., items/interactions), the modalities (i.e., visual/textual/audio), and the list of models along with output layers and preprocessing steps to extract the features.

## How can I run Ducho

You may choose among three options:

- Locally by cloning this GitHub repo.
- By pulling our docker image on Docker Hub ([link](https://hub.docker.com/repository/docker/sisinflabpoliba/ducho/general)).
- On Google Colab ([link](https://colab.research.google.com/drive/1ouKkdxOObOL0BI00r0c157oNRqwxqTgt)).

## Prerequisites

### Local

Ducho may work on both CPU and GPU. However, if you want to speed up your feature extraction, we highly recommend to go for the GPU-accelerated option. 

In that case, if your machine is equipped with NVIDIA drivers, you should first make sure CUDA is installed along with the compatible NVIDIA drivers. 

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

```sh
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
meaning that the installation is ok and you are finally ready to pull ducho's image (which is actually built from this CUDA image)!

### Google Colab

You just need a Google Drive account!

## Installation

Depending on where you are running Ducho, you might need to first clone this repo and install the necessary packages.

If you are running Ducho on your local machine or Google Colab, you first need to clone this repository:

```sh
git clone https://github.com/sisinflab/Ducho.git
```

Then, install the needed dependencies through pip:

```sh
pip install -r requirements.txt # Local
pip install -r requirements_colab.txt # Google Colab
```

P.S. Since Google Colab already comes with almost all necessary packages, you just need to install very few missing ones.

Note that these two steps are not necessary for the docker version because the image already comes with the suitable environment.

## Try Ducho

To ease the usage of Ducho, here we provide three demos spanning different multimodal recommendation scenarios. Use them to better familiarize with the framework:

- **Demo 1:** visual and textual feature extraction from items ([link](demos/demo1/README.md))
- **Demo 2:** audio and textual feature extraction from items ([link](demos/demo2/README.md))
- **Demo 3:** textual feature extraction from items and interactions ([link](demos/demo3/README.md))

## Use Ducho

Once you have familiarized with Ducho, you can use it for your own datasets and custom multimodal feature extractions! Please refer to the official [documentation](https://ducho.readthedocs.io/en/latest/) where all modules, classes, and methods are explained in detail.

You may also consider to take a look at this [guideline](config/README.md) to better understand how to fill in your custom configuration files.


### Overwrite Configuration trought command
To change an entry in the Configuration dict, you need to write the changes in a list that will be given to the Runner with the key argv.
This list is composed of tuples key-value where each entry is a String that needs to be write in this way: ’key=value’. As the config dictionary is built of other nested dictionaries, the key is used also to describe the path within to follow, to do so the ’.’ is used to discern. I.e ’key1.key2.key3’
