import torchaudio
import torch
import numpy as np
from ducho.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather
from transformers import Wav2Vec2Model


class AudioCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Audio Extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        Args:
            gpu: String on which is explained which gpu to use. '-1' -> cpu
        """
        self._model_to_initialize = None
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model_name: is the name of the model to use.
        Returns: nothing but it initializes the protected model, later used for extraction
        :param model:
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
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        Args:
            sample_input: the sample preprocessed within the dataset class

        Returns: a numpy array that will be put in a .npy file calling the right Dataset Class' method

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
