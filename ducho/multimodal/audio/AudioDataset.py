import os
import re
import numpy
import torchaudio
from ducho.internal.father_classes.DatasetFather import DatasetFather
from ducho.internal.utils.TextualFileManager import TextualFileManager
import soundfile
from transformers import Wav2Vec2Processor


class AudioDataset(DatasetFather):
    """
    This class represents the Audio Dataset used for the data loading process.
    """
    def __init__(self, input_directory_path, output_directory_path):
        """
        It manages the Audio Dataset, which consists of a folder containing input data and another folder for output data.
        It handles the preprocessing of input data and manages the output data.

        Args:
            input_directory_path: folder of the input data to elaborate as String
            output_directory_path: folder of where put Output as String, it will be created if it does not exist

        Returns:
            None
        """
        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._model_for_preprocessing = None

    def __getitem__(self, index):
        """
        It retrieves a sample preprocessed given its id (the id refers to the sorted filenames).

        Args:
            index: Integer, indicates the number associated to the file o elaborate.

        Returns:
             the audio resembled (preprocessed) and its sample rate. [audio, rate]
        """
        audio_path = os.path.join(self._input_directory_path, self._filenames[index])

        # right now both torchaudio and transformers do the same thing, but it is preferable to keep them work
        # separately for future improvement

        if 'torch' in self._backend_libraries_list or 'torchaudio' in self._backend_libraries_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate]), None
        elif 'transformers' in self._backend_libraries_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate]), None

    def set_model(self, model):
        """
        sets the model as a string to execute the preprocessing
        NOTE ON MODELS:
        here it is accepted torchaudio and transformers (by huggingface) models. When using transformers you have to
        indicate in the String also the repo as 'repo/model_name'

        Args:
             model: the model name as a String

        Returns:
            None
        """
        self._model_name = model

    def _pre_processing(self, pre_process_input):
        """
        It resample the audio to a rate that is the same to the one with the models where trained.

        Args:
            pre_process_input: blob of data (audio wave and audio rate).

        Returns:
            [preprocessed audio, new sample rate]
        """
        audio = pre_process_input[0]
        rate = pre_process_input[1]
        if 'torch' in self._backend_libraries_list or 'torchaudio' in self._backend_libraries_list:
            bundle = getattr(torchaudio.pipelines, self._model_name)
            waveform = torchaudio.functional.resample(audio, rate, bundle.sample_rate)
            return [waveform, bundle.sample_rate]
        elif 'transformers' in self._backend_libraries_list:
            pre_processor = torchaudio.transforms.Resample(rate, 16000)
            resampled_audio = pre_processor(audio)
            return [resampled_audio, 16000]

    def set_preprocessing_flag(self, preprocessing_flag):
        return
