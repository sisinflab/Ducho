import tensorflow as tf
import torch
import numpy as np
# import torchvision.models import
# from torchvision import
import torchvision
import tensorflow
# from torchvision.models import ResNet50_Weights


from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather
import sys


class VisualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Image Extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        :param gpu:
        """
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model_name: is the name of the model to use.
        Returns: nothing but it initializes the protected model attribute, later used for extraction
        :param model:
        """
        model_name = model['name']
        torchvision_list = list(torchvision.models.__dict__)
        tensorflow_keras_list = list(tensorflow.keras.applications.__dict__)

        self._model_name = model_name
        if self._model_name in tensorflow_keras_list and 'tensorflow' in self._framework_list:
            self._model = getattr(tensorflow.keras.applications, self._model_name)()
        elif self._model_name.lower() in torchvision_list and 'torch' in self._framework_list:
            self._model = getattr(torchvision.models, self._model_name.lower())(weights='DEFAULT')
            self._model.to(self._device)
            self._model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_feature(self, image):
        """
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        :param image: the data of the image preprocessed
        :return: a numpy array that will be put in a .npy file calling the right Dataset Class' method
        """
        torchvision_list = list(torchvision.models.__dict__)
        if self._model_name.lower() in torchvision_list and 'torch' in self._framework_list:
            # torch
            if isinstance(list(self._model.children())[-1], torch.nn.Linear):
                s1 = torch.nn.Sequential(*list(self._model.children())[:-self._output_layer])
                s2 = torch.nn.Flatten()
                feature_model = torch.nn.Sequential(s1, s2)
            else:
                s1 = torch.nn.Sequential(*list(self._model.children())[:-1])
                s2 = torch.nn.Flatten()
                s3 = torch.nn.Sequential(*list(list(self._model.children())[-1][1:-self._output_layer]))
                feature_model = torch.nn.Sequential(s1, s2, s3)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self._device)
            ).data.cpu().numpy())
            # update the framework list
            self._framework_list = ['torch']
        else:
            # tensorflow
            input_model = self._model.input
            output_layer = self._model.get_layer(self._output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)
            # update the framework list
            self._framework_list = ['tensorflow']

        return output
