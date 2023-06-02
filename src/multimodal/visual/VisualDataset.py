from abc import ABC

from PIL import Image
from src.internal.father_classes.DatasetFather import DatasetFather
from torchvision import transforms
import tensorflow
import numpy as np
import os


class VisualDataset(DatasetFather, ABC):

    def __init__(self, input_directory_path, output_directory_path, model_name='VGG19', reshape=(224, 224)):
        """
        Manage the Image Dataset (folder of input and folder of output).
        It will Manage data of input (and their preprocessing), and data of output
        :param input_directory_path: folder of the input data to elaborate as String
        :param output_directory_path: folder of where put Output as String, it will be created if does not exist
        :param model_name: String of the model to use, it can be reset later
        :param reshape: Tuple (int, int), is width and height for the resize, it can be reset later
        """
        super().__init__(input_directory_path, output_directory_path, model_name)
        self._reshape = reshape

    # def set_model_map(self, model_map_path):
    # print(model_map_path)

    def __getitem__(self, idx):
        """
        It retrieves a sample preprocessed given its id (the id refers to the sorted filenames)
        :param idx: Integer, indicates the number associated to the file o elaborate
        :return: the image blob data preprocessed
        """
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self._pre_processing(sample)

        if 'tensorflow' in self._framework_list:
            # np for tensorflow
            return np.expand_dims(norm_sample, axis=0)
        else:
            # torch
            return norm_sample

    def _pre_processing(self, sample):
        """
        It prepares the data to the feature extraction
        :param sample: the image just read
        :return: the image resized and normalized
        """
        # resize
        if self._reshape:
            res_sample = sample.resize(self._reshape, resample=Image.BICUBIC)
        else:
            res_sample = sample

        # normalize
        tensorflow_keras_list = list(tensorflow.keras.applications.__dict__)
        if self._model_name.lower() in tensorflow_keras_list and 'tensorflow' in self._framework_list:
            # if the model is a tensorflow model, each one execute a different command (retrieved from the model map)
            # command_two = tensorflow_models_for_normalization[self._model_name]
            command = getattr(tensorflow.keras.applications, self._model_name.lower())
            norm_sample = command.preprocess_input(np.array(res_sample))
            # update the framework list
            self._framework_list = ['tensorflow']
        else:
            # if the model is a torch model, the normalization is the same for everyone
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
            norm_sample = transform(res_sample)
            # update the framework list
            self._framework_list = ['torch']

        return norm_sample

    def set_reshape(self, reshape):
        """
        Set the reshape data to reshape the image (resize)
        :param reshape: Tuple (int, int), is width and height
        """
        self._reshape = reshape

    def set_preprocessing_flag(self, preprocessing_flag):
        self._reshape = preprocessing_flag
