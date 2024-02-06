Introduction
======================

Ducho v2.0 is a Python framework for the extraction of multimodal features for recommendation. It provides a unified interface to most of the common libraries for deep learning (e.g., TensorFlow, PyTorch, Transformers, Sentence-Transformers) to extract high-level features from items (e.g., product images/descriptions) and user-item interactions (e.g., users reviews). It is highly configurable through a YAML-based configuration file (which may be override by input arguments from the command line in case). Users can indicate the source from which to extract the multimodal features (i.e., items/interactions), the modalities (i.e., visual/textual/audio/multiple), and the list of models along with output layers and preprocessing steps to extract the features. Moreover, with the new version of Ducho, users can conduct extractions by utilizing their own pretrained models.


For all other details about Ducho please refer to our `paper <https://google.it>`_ and cite [Ducho]_


.. [Ducho]
    Matteo Attimonelli and Danilo Danese Daniele Malitesta and Giuseppe Gassi and Claudio Pomo and Tommaso Di Noia

    Ducho 2.0: Towards a More Up-to-Date Feature Extraction and Processing Framework for Multimodal Recommendation
