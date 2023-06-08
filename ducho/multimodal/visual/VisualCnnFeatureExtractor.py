import tensorflow as tf
import numpy as np
import torchvision
import tensorflow
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from ducho.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


class VisualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Image Extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        Args:
             gpu:
        """
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model: is the model to use.
        Returns:
            nothing but it initializes the protected model attribute, later used for extraction

        """
        model_name = model['name']
        torchvision_list = list(torchvision.models.__dict__)
        tensorflow_keras_list = list(tensorflow.keras.applications.__dict__)

        self._model_name = model_name
        if self._model_name in tensorflow_keras_list and 'tensorflow' in self._backend_libraries_list:
            self._model = getattr(tensorflow.keras.applications, self._model_name)()
        elif self._model_name.lower() in torchvision_list and 'torch' in self._backend_libraries_list:
            self._model = getattr(torchvision.models, self._model_name.lower())(weights='DEFAULT')
            self._model.to(self._device)
            self._model.eval()
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_feature(self, image):
        """
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        Args:
            image: the data of the image preprocessed
        Returns:
             a numpy array that will be put in a .npy file calling the right Dataset Class' method
        """
        torchvision_list = list(torchvision.models.__dict__)
        if self._model_name.lower() in torchvision_list and 'torch' in self._backend_libraries_list:
            _, eval_nodes = get_graph_node_names(self._model)
            return_nodes = {}
            output_layer = 'layer0'
            for idx, e in enumerate(eval_nodes):
                return_nodes[e] = f'layer{idx}'
                if e == self._output_layer:
                    output_layer = f'layer{idx}'
                    break
            feature_model = create_feature_extractor(self._model, return_nodes)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image[None, ...].to(self._device)
            )[output_layer].data.cpu().numpy())
            # update the framework list
            self._backend_libraries_list = ['torch']
        else:
            # tensorflow
            input_model = self._model.input
            output_layer = self._model.get_layer(self._output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)
            # update the framework list
            self._backend_libraries_list = ['tensorflow']
        return output
