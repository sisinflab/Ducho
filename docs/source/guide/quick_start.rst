How can I run Ducho
------------

You may choose among three options:

- Locally by cloning this GitHub repo.
- By pulling our docker image on Docker Hub (`link <https://hub.docker.com/repository/docker/sisinflabpoliba/ducho/general>`_).
- On Google Colab (`link <https://colab.research.google.com/drive/1vPUALePlrjv4rfSn6CX2zMkpH2Xrw_cp>`_).

Prerequisites
^^^^^^^^^^^^^^^^

Local
^^^^^^^^^

Ducho may work on both CPU and GPU, harnessing the power of CUDA and MPS engines. However, if you want to speed up your feature extraction, we highly recommend to go for the GPU-accelerated option.

In that case, if your machine is equipped with NVIDIA drivers, you should first make sure CUDA is installed along with the compatible NVIDIA drivers.

For example, a possible working environment involves the following (you may refer to any Google Colab notebook):

.. code:: bash
    Nvidia drivers: 525.85.12
    Cuda: 11.8.89
    Python: 3.11.2
    Pip: 23.3.2


Please, refer to this `link <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ for the official guidelines on how to install the nvidia toolkit on linux from scratch.

Docker
^^^^^^^^^

Docker is easily the best option to run any NVIDIA/CUDA-based framework, because it provides docker images with everything already setup for almost all your needs.

First of all, you need to install the docker engine on your machine. Here is the official `link <https://docs.docker.com/engine/install/ubuntu/>`_ for ubuntu. Then, for the sake of the demos provided in this repository, you might also need docker compose (here is a reference `link <https://docs.docker.com/compose/install/standalone/>`_).

Quite conveniently, you can find several CUDA-equipped images on Docker Hub. You may refer to this `link <https://hub.docker.com/r/nvidia/cuda>`_. Depending on your CUDA version, you may also need to install nvidia-docker2 (in case, here is a `reference <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_).

To test if everything worked smoothly, pull and run a container from the following docker image through this command:

.. code:: bash
    docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 nvidia-smi


Once the docker image has been downloaded from the hub, you should be able to see something like this:

meaning that the installation is ok and you are finally ready to pull ducho's image (which is actually built from this CUDA image)!

Google Colab
^^^^^^^^^^^^^^^

You just need a Google Drive account!

