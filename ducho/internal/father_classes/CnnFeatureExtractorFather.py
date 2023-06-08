import torch
import tensorflow as tf
import numpy as np


class CnnFeatureExtractorFather:
    def __init__(self, gpu='-1'):
        self._backend_libraries_list = None
        self._model = None
        self._output_layer = None
        self._model_name = None
        self._gpu = gpu

        self._device = torch.device(f'cuda:{self._gpu}' if torch.cuda.is_available() else 'cpu')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    def set_output_layer(self, output_layer):
        self._output_layer = output_layer

    def set_framework(self, backend_libraries_list):
        """
        It set the framework to use as e.g: 'torch', 'tensorflow', 'transformers', 'torchaudio'
        Args:
            backend_libraries_list: the list of String of the framework. It's acceptable to have only one item in the list

        """
        self._backend_libraries_list = backend_libraries_list

    def set_model(self, model_name):
        pass

    def extract_feature(self, image):
        pass
