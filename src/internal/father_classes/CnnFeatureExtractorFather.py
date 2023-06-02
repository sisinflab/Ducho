import tensorflow as tf
import torch
import numpy as np


class CnnFeatureExtractorFather:
    def __init__(self, gpu='-1'):
        self._framework_list = None
        self._model = None
        self._output_layer = None
        self._model_name = None
        self._gpu = gpu

        self._device = torch.device("cuda:" + str(self._gpu) if self._gpu != '-1' else "cpu")

    def set_output_layer(self, output_layer):
        self._output_layer = output_layer

    def set_framework(self, framework_list):
        """
        It set the framework to use as e.g: 'torch', 'tensorflow', 'transformers', 'torchaudio'
        :param framework_list: the list of String of the framework. It's acceptable to have only one item in the list
        """
        self._framework_list = framework_list

    def set_model(self, model_name):
        pass

    def extract_feature(self, image):
        pass
