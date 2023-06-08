Introduction
======================

Ducho is a Python framework for the extraction of multimodal features for recommendation. It provides a unified interface to most of the common libraries for deep learning (e.g., TensorFlow, PyTorch, Transformers) to extract high-level features from items (e.g., product images/descriptions) and user-item interactions (e.g., users reviews). It is highly configurable through a YAML-based configuration file (which may be override by input arguments from the command line in case). Users can indicate the source from which to extract the multimodal features (i.e., items/interactions), the modalities (i.e., visual/textual/audio), and the list of models along with output layers and preprocessing steps to extract the features.



For all other details about Ducho please refer to our `paper <https://google.it>`_ and cite [Ducho]_


.. [Ducho]
    Daniele Malitesta and Giuseppe Gassi and Claudio Pomo and Tommaso Di Noia

    Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation
