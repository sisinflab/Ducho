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

### Docker

### Google Colab

You just need a google drive account!
