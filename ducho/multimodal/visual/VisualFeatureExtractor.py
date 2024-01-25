import tensorflow as tf
import numpy as np
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather
from transformers import pipeline


class VisualFeatureExtractor(FeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Image Extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        Args:
             gpu:
        """
        self._pipeline = None
        self._image_processor = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model: is the model to use.
        Returns:
            nothing but it initializes the protected model attribute, later used for extraction

        """
        model_name = model['model_name']
        image_processor = model['image_processor'] if 'image_processor' in model else None
        torchvision_list = list(torchvision.models.__dict__)
        tensorflow_keras_list = list(tf.keras.applications.__dict__)

        self._model_name = model_name
        if self._model_name in tensorflow_keras_list and 'tensorflow' in self._backend_libraries_list:
            self._model = getattr(tf.keras.applications, self._model_name)()
        elif self._model_name.lower() in torchvision_list and 'torch' in self._backend_libraries_list:
            self._model = getattr(torchvision.models, self._model_name.lower())(weights='DEFAULT')
            self._model.to(self._device)
            self._model.eval()
        elif 'torch' in self._backend_libraries_list:
            # Custom Model Loading
            self._model = torch.load(model_name, map_location=self._device)
        elif 'transformers' in self._backend_libraries_list:
            built_pipeline = pipeline(task='feature-extraction', model=model_name, image_processor=image_processor, framework='pt', device=self._device)
            self._model = built_pipeline.model
            self._image_processor = built_pipeline.image_processor
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
        tensorflow_keras_list = list(tf.keras.applications.__dict__)
        if 'torch' in self._backend_libraries_list: #and self._model_name.lower() in torchvision_list:
            _, eval_nodes = get_graph_node_names(self._model)
            return_nodes = {}
            output_layer = 'layer0'
            found = False
            for idx, e in enumerate(eval_nodes):
                return_nodes[e] = f'layer{idx}'
                if e == self._output_layer:
                    output_layer = f'layer{idx}'
                    found = True
                    break
            if not found:
                raise ValueError(f"The specified output layer {self._output_layer} does not exist. Please carefully check its name!")
            feature_model = create_feature_extractor(self._model, return_nodes)
            feature_model.eval()
            output = np.squeeze(feature_model(
                image.to(self._device)
            )[output_layer].data.cpu().numpy())
            # update the framework list
            self._backend_libraries_list = ['torch']
            return output
        elif self._model_name in tensorflow_keras_list and 'tensorflow' in self._backend_libraries_list:
            # tensorflow
            input_model = self._model.input
            output_layer = self._model.get_layer(self._output_layer).output
            output = tf.keras.Model(input_model, output_layer)(image, training=False)
            # update the framework list
            self._backend_libraries_list = ['tensorflow']
            return output
        elif 'transformers' in self._backend_libraries_list:
            model_input = self._image_processor(image, return_tensors="pt", do_rescale=False)
            model_input = {k: torch.tensor(v).to(self._device) for k, v in model_input.items()}
            model_output = getattr(self._model(**model_input), self._output_layer.lower())
            return model_output.detach().cpu().numpy()
