import torchaudio
import torch
import numpy as np
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather
from transformers import Wav2Vec2Model


class AudioFeatureExtractor(FeatureExtractorFather):
    """
        This class represents the Audio Feature Extractor utilized for feature extraction.
    """
    def __init__(self, gpu='-1'):
        """
        This function carries out Audio Feature Extraction, requiring the 'model_name', 'framework', and 'output_layer'.

        Args:
            gpu: A string indicating the GPU to be used. '-1' specifies the CPU.

        Returns:
            None
        """
        self._model_to_initialize = None
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        This procedure facilitates the configuration of the Audio Feature Extractor model using YAML specifications.

        Args:
            model: The row of the YAML file containing the user's specifications.

        Returns:
            None
        """
        model_name = model['name']
        self._model_name = model_name
        if 'torch' in self._backend_libraries_list or 'torchaudio' in self._backend_libraries_list:
            model_to_initialize = getattr(torchaudio.pipelines, model_name)
            self._model = model_to_initialize.get_model()
            self._model.to(self._device)
            self._model.eval()
            # self._model.to(self._gpu)
        elif 'transformers' in self._backend_libraries_list:
            self._model = Wav2Vec2Model.from_pretrained(self._model_name)

    def extract_feature(self, sample_input):
        """
        This function extracts features from the input data. Prior to calling this function, the framework,
        model, and layer have to be configured using their respective set methods.

        Args:
            sample_input: The preprocessed data.

        Returns:
            A numpy array representing the extracted features, which will be stored in a .npy file using the appropriate method of the Dataset Class.
        """
        audio = sample_input[0]
        sample_rate = sample_input[1]
        if 'torch' in self._backend_libraries_list or 'torchaudio' in self._backend_libraries_list:
            # extraction
            # num_layer is the number of layers to go through
            try:
                features, _ = self._model.extract_features(audio, num_layers=self._output_layer)
                feature = features[-1]
                # return the N-Dimensional Tensor as a numpy array
                return feature.detach().numpy()
            except AttributeError:
                if isinstance(list(self._model.children())[-1], torch.nn.Linear):
                    feature_model = torch.nn.Sequential(*list(self._model.children())[:-self._output_layer])
                else:
                    feature_model = self._model
                feature_model.eval()
                output = np.squeeze(feature_model(
                    audio[None, ...].to(self._device)
                ).data.cpu().numpy())
                # update the framework list
                self._backend_libraries_list = ['torch']
                return output

        elif 'transformers' in self._backend_libraries_list:
            # feature extraction
            outputs = self._model(audio, output_hidden_states=True)
            # layer extraction
            layer_output = outputs.hidden_states[self._output_layer]
            return layer_output.detach().numpy()
