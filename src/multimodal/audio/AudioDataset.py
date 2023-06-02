import os
import re
import numpy
import torchaudio
from src.internal.father_classes.DatasetFather import DatasetFather
from src.internal.utils.TextualFileManager import TextualFileManager
import soundfile
from transformers import Wav2Vec2Processor


class AudioDataset(DatasetFather):

    def __init__(self, input_directory_path, output_directory_path):
        """
        Manage the Audio Dataset (folder of input and folder of output).
        It will Manage data of input (and their preprocessing), and data of output
        :param input_directory_path: folder of the input data to elaborate as String
        :param output_directory_path: folder of where put Output as String, it will be created if does not exist
        """
        super().__init__(input_directory_path, output_directory_path, model_name=None)
        self._model_for_preprocessing = None

    def __getitem__(self, index):
        """
        It retrieves a sample preprocessed given its id (the id refers to the sorted filenames)
        :param index: Integer, indicates the number associated to the file o elaborate
        :return: the audio resembled (preprocessed) and its sample rate. [audio, rate]
        """
        audio_path = os.path.join(self._input_directory_path, self._filenames[index])

        # right now both torchaudio and transformers do the same thing, but it is preferable to keep them work
        # separately for future improvement

        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate])
        elif 'transformers' in self._framework_list:
            audio, sample_rate = torchaudio.load(audio_path)
            return self._pre_processing([audio, sample_rate])

    def set_model(self, model):
        """
        sets the model as a string to execute the preprocessing
        NOTE ON MODELS:
        here it is accepted torchaudio and transformers (by huggingface) models. When using transformers you have to
        indicate in the String also the repo as 'repo/model_name'
        :param model: the model name as a String
        """
        self._model_name = model

    def _pre_processing(self, pre_process_input):
        """
        It resample the audio to a rate that is the same to the one with the models where trained
        :param pre_process_input: blob of data (audio wave and audio rate )
        :return: [preprocessed audio, new sample rate]
        """
        audio = pre_process_input[0]
        rate = pre_process_input[1]
        if 'torch' in self._framework_list or 'torchaudio' in self._framework_list:
            bundle = getattr(torchaudio.pipelines, self._model_name)
            waveform = torchaudio.functional.resample(audio, rate, bundle.sample_rate)
            return [waveform, bundle.sample_rate]
        elif 'transformers' in self._framework_list:
            pre_processor = torchaudio.transforms.Resample(rate, 16000)
            resampled_audio = pre_processor(audio)
            return [resampled_audio, 16000]

    def set_preprocessing_flag(self, preprocessing_flag):
        return
