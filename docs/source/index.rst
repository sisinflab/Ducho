.. Ducho documentation master file, created by
   sphinx-quickstart on Tue May 30 17:32:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Ducho's documentation!
=================================

What is Ducho
-------------

Ducho is a Python framework for the extraction of multimodal features for recommendation. It provides a unified
interface to most of the common libraries for deep learning (e.g., TensorFlow, PyTorch, Transformers) to extract
high-level features from items (e.g., product images/descriptions) and user-item interactions (e.g., users reviews).
It is highly configurable through a YAML-based configuration file (which may be override by input arguments from the
command line in case). Users can indicate the source from which to extract the multimodal features
(i.e., items/interactions), the modalities (i.e., visual/textual/audio), and the list of models along with output
layers and preprocessing steps to extract the features.

.. figure:: ./img/ducho_img.png
   :alt: system schema

   system schema


.. flat-table:: Characteristics of the BLE badge
   :header-rows: 2

   * -
     - :cspan:`1` Sources
     - :cspan:`2` Backends
   * - Modalities
     - Items
     - Interactions
     - .. image:: https://www.gstatic.com/devrel-devsite/prod/v37463d4834445c1e880de1e91d2f8fc2c6a0e86fca4aa6a7bdbb270b040181dc/tensorflow/images/lockup.svg
          :height: 20 px
          :width: 90 px
          :alt: tensorflow
          :align: center
     - .. image:: https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png
          :height: 15 px
          :width: 62 px
          :alt: Pytorch_logo
          :align: center
     - .. image:: https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png
          :height: 16 px
          :width: 99 px
          :alt: huggingface
          :align: center
   * - Audio
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
     -
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
   * - Visual
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
     -
   * - Text
     - .. image:: ./img/checkbox-mark.png
          :align: center
     - .. image:: ./img/checkbox-mark.png
          :align: center
     -
     -
     - .. image:: ./img/checkbox-mark.png
          :align: center


.. toctree::
   :maxdepth: 1
   :caption: GET STARTED

   guide/introduction
   guide/install
   guide/quick_start


.. toctree::
   :maxdepth: 1
   :caption: API REFERENCE:

   ducho/ducho
   ducho/ducho.config
   ducho/ducho.internal
   ducho/ducho.multimodal
   ducho/ducho.runner

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
