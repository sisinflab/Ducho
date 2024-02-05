import torch
import tensorflow as tf
import numpy as np


class FeatureExtractorFather:
    """
    Represents a feature extractor object.

    This class provides functionality for feature extraction using various backend libraries and models.

    Attributes:
        _backend_libraries_list: A list of backend libraries (e.g. tensorflow, pytorch, etc.).
        _model: The model for feature extraction.
        _output_layer: The output layer for feature extraction.
        _model_name: The name of the model.
        _gpu (str): The GPU index or '-1' for CPU.
        _device: The device for computation (GPU, MPS, CPU).
    """
    def __init__(self, gpu='-1'):
        """
        Initialize the FeatureExtractorFather object.

        Args:
            gpu (str, optional): The GPU index or '-1' for CPU.

        Returns:
            None
        """
        self._backend_libraries_list = None
        self._model = None
        self._output_layer = None
        self._model_name = None
        self._gpu = gpu

        self._device = torch.device(f'cuda:{self._gpu}' if torch.cuda.is_available() else 'cpu')
        self._device = torch.device(f'mps' if torch.backends.mps.is_available() else self._device)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def set_output_layer(self, output_layer):
        self._output_layer = output_layer

    def set_framework(self, backend_libraries_list):
        """
        Set the framework(s) for use (e.g. tensorflow, pytorch, etc.).

        Args:
            backend_libraries_list (List[str]): A list of strings representing the framework(s) to utilize.
                It is acceptable to have only one item in the list.

        Returns:
            None

        """
        self._backend_libraries_list = backend_libraries_list

    def set_model(self, model_name):
        pass

    def extract_feature(self, image):
        pass
