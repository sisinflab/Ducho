from abc import ABC

from PIL import Image
from ducho.internal.father_classes.DatasetFather import DatasetFather
from torchvision import transforms
import tensorflow
import numpy as np
import os
import torch


class MinMaxNormalize(object):
    def __call__(self, img):

        min_value = img.min()
        max_value = img.max()

        normalized_img = (img - min_value) / (max_value - min_value)

        return normalized_img


class VisualDataset(DatasetFather, ABC):

    def __init__(self, input_directory_path, output_directory_path, model_name='VGG19', reshape=(224, 224)):
        """
        Manage the Image Dataset (folder of input and folder of output).
        It will Manage data of input (and their preprocessing), and data of output
        Args:
            input_directory_path: folder of the input data to elaborate as String
            output_directory_path: folder of where put Output as String, it will be created if does not exist
            model_name: String of the model to use, it can be reset later
            reshape: Tuple (int, int), is width and height for the resize, it can be reset later
        """
        super().__init__(input_directory_path, output_directory_path, model_name)
        self._reshape = reshape
        self._preprocessing_type = None
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]

    # def set_model_map(self, model_map_path):
    # print(model_map_path)

    def __getitem__(self, idx):
        """
        It retrieves a sample preprocessed given its id (the id refers to the sorted filenames)
        Args:
            idx: Integer, indicates the number associated to the file o elaborate
        Returns:
             the image blob data preprocessed
        """
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self._pre_processing(sample)

        if 'tensorflow' in self._backend_libraries_list:
            # np for tensorflow
            return np.expand_dims(norm_sample, axis=0), None
        else:
            # torch
            return norm_sample, None

    def _pre_processing(self, sample):
        """
        It prepares the data to the feature extraction
        Args:
            sample: the image just read
        Returns:
             the image resized and normalized
        """
        # resize
        if self._reshape:
            res_sample = sample.resize(self._reshape, resample=Image.BICUBIC)
        else:
            res_sample = sample

        # normalize
        tensorflow_keras_list = list(tensorflow.keras.applications.__dict__)
        if self._model_name.lower() in tensorflow_keras_list and 'tensorflow' in self._backend_libraries_list:
            # if the model is a tensorflow model, each one execute a different command (retrieved from the model map)
            # command_two = tensorflow_models_for_normalization[self._model_name]
            command = getattr(tensorflow.keras.applications, self._model_name.lower())
            norm_sample = command.preprocess_input(np.array(res_sample))
            # update the framework list
            self._backend_libraries_list= ['tensorflow']
        else:
            # if the model is a torch model, the normalization is the same for everyone
            # print(self._preprocessing_type)
            if self._preprocessing_type is not None:
                if self._preprocessing_type == 'zscore':
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(mean=self._mean,
                                                                         std=self._std)
                                                    ])
                else:
                    
                    transform = transforms.Compose([transforms.ToTensor(),
                                                    MinMaxNormalize()
                                                    ])
            else:
                transform = transforms.ToTensor()
            
            norm_sample = transform(res_sample)

            # update the framework list
            self._backend_libraries_list = ['torch']
        
        return norm_sample

    def set_reshape(self, reshape):
        """
        Set the reshape data to reshape the image (resize)
        Args:
             reshape: Tuple (int, int), is width and height
        """
        self._reshape = reshape

    def set_preprocessing_flag(self, preprocessing_flag):
        self._reshape = preprocessing_flag

    def set_preprocessing_type(self, 
                               preprocessing_type: str
                               ) -> None:
        """
        Set the desired pre-processing type between minmax and z-score
        Args:
             mean: torch.Tensor containing the desired mean along the three channels
             std: torch.Tensor containing the desired standard deviation along the three channels
        """
        self._preprocessing_type = preprocessing_type

    def set_mean_std(self, 
                     mean: torch.Tensor, 
                     std: torch.Tensor
                     ) -> None:
        """
        Set custom values of mean and std for z-score normalization
        Args:
             mean: torch.Tensor containing the desired mean along the three channels
             std: torch.Tensor containing the desired standard deviation along the three channels
        """
        self._mean = mean
        self._std = std

    def _reset_mean_std(self) -> None:
        """
        Reset mean and std values to ImageNet ones
        """
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]