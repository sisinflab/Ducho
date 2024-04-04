import tensorflow as tf
import numpy as np
import torchvision
import torch
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather
from transformers import pipeline


class VisualFeatureExtractor(FeatureExtractorFather):
    """
    This class represents the Visual Feature Extractor utilized for feature extraction.
    """
    def __init__(self, gpu='-1'):
        """
        This function carries out Image Feature Extraction, requiring the 'model_name', 'framework', and 'output_layer'.

        Args:
             gpu: A string indicating the GPU to be used. '-1' specifies the CPU.

        Returns:
            None
        """
        self._pipeline = None
        self._image_processor = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        This procedure facilitates the configuration of the Visual Feature Extractor model using YAML specifications.

        Args:
            model: The row of the YAML file containing the user's specifications.

        Returns:
            None
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
            raise NotImplementedError('This feature extractor has not been added yet!')

    @torch.no_grad
    def extract_feature(self, image):
        """
        This function extracts features from the input image data. Prior to calling this function, the framework,
        model, and layer have to be configured using their respective set methods.

        Args:
            image: The preprocessed image data.

        Returns:
            A numpy array representing the extracted features, which will be stored in a .npy file using the appropriate method of the Dataset Class.
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
            with torch.no_grad():
                output = np.squeeze(feature_model(
                    image[0].to(self._device)
                )[output_layer].data.cpu().numpy())
            # update the framework list
            self._backend_libraries_list = ['torch']
            return np.expand_dims(output, axis=0) if output.ndim == 1 else output
        
        elif self._model_name in tensorflow_keras_list and 'tensorflow' in self._backend_libraries_list:
            # tensorflow
            input_model = self._model.input
            output_layer = self._model.get_layer(self._output_layer).output
            output = np.array(tf.keras.Model(input_model, output_layer)(image[0][None], training=False))
            # update the framework list
            self._backend_libraries_list = ['tensorflow']
            return np.expand_dims(output, axis=0) if output.ndim == 1 else output
        
        elif 'transformers' in self._backend_libraries_list:
            # converting the input image tensor - outcome of the pre-processor - in a set.
            model_input = {'pixel_values': image[0]}
            model_input = {k: torch.tensor(v).to(self._device) for k, v in model_input.items()}
            model_output = getattr(self._model(**model_input), self._output_layer.lower())
            output = model_output.detach().cpu().numpy()
            return np.expand_dims(output, axis=0) if output.ndim == 1 else output
