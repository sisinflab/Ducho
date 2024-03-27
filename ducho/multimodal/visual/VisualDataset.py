from abc import ABC

from PIL import Image
from ducho.internal.father_classes.DatasetFather import DatasetFather
from torchvision import transforms
import tensorflow
import numpy as np
import os
import torch


class MinMaxNormalize(object):
    """
    This class allows to perform the MinMAx normalization.
    """
    def __call__(self, img):
        """
        This method yields the image preprocessed through MinMax normalization.

        Args:
            img: The input image

        Returns:
            The normalized image
        """
        min_value = img.min()
        max_value = img.max()

        normalized_img = (img - min_value) / (max_value - min_value)

        return normalized_img


class VisualDataset(DatasetFather, ABC):
    """
    This class represents the Visual Dataset used for the data loading process.
    """
    def __init__(self, input_directory_path, output_directory_path, model_name='VGG19', reshape=(224, 224)):
        """
        It manages the Image Dataset, which consists of a folder containing input data and another folder for output data.
        It handles the preprocessing of input data and manages the output data.

        Args:
            input_directory_path: A string representing the path to the folder containing the input data to be processed.
            output_directory_path: A string representing the path to the folder where the output data will be stored. If the folder does not exist, it will be created.
            model_name: A string specifying the model to be used. This can be reset later.
            reshape: A tuple (int, int) representing the width and height for resizing the input images. This can be reset later.

        Returns:
            None
        """
        super().__init__(input_directory_path, output_directory_path, model_name)
        self._reshape = reshape
        self._preprocessing_type = None
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
        
        # _image_processor is needed for the transformers' library, otherwise it must be None.
        self._image_processor = None

    def __getitem__(self, idx):
        """
        It retrieves a sample preprocessed given its id (the id refers to the sorted filenames).

        Args:
            idx: Integer, indicates the number associated to the file o elaborate.

        Returns:
             the image blob data preprocessed.
        """
        image_path = os.path.join(self._input_directory_path, self._filenames[idx])
        sample = Image.open(image_path)

        if sample.mode != 'RGB':
            sample = sample.convert(mode='RGB')

        norm_sample = self._pre_processing(sample)

        if 'tensorflow' in self._backend_libraries_list:
            # np for tensorflow
            return np.expand_dims(norm_sample, axis=0)
        else:
            # torch
            return norm_sample, self._filenames[idx]

    def _pre_processing(self, sample):
        """
        It pre-process the data for the feature extraction.

        Args:
            sample: the read image.

        Returns:
             the processed image.
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
            command = getattr(tensorflow.keras.applications, self._model_name.lower())
            norm_sample = command.preprocess_input(np.array(res_sample))
            # update the framework list
            self._backend_libraries_list = ['tensorflow']
        elif 'torch' in self._backend_libraries_list:
            # if the model is a torch model, the normalization is the same for everyone
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
            self._backend_libraries_list = ['torch']

        elif 'transformers' in self._backend_libraries_list:
            # transformers' visual features extractor inputs need to be properly preprocessed with the respective pre-processors. 
            # aiming to avoid dataloader issues, norm_sample contains a torch.tensor, instead of a set.
            norm_sample = self._image_processor(res_sample).pixel_values[0]
            # update the framework list
            self._backend_libraries_list = ['transformers']

        return norm_sample

    def set_reshape(self, reshape):
        """
        Set the reshape variable according to the desired value.

        Args:
             reshape: Tuple (int, int) representing the width and height for resizing the input.

        Returns:
            None
        """
        self._reshape = reshape

    def set_image_processor(self, image_processor):
        """
        Set the image_processor functional pointer for the tranformers library.
        Args:
            image_processor: the image processor function.

        Returns:
            None
        """
        assert 'transformers' in self._backend_libraries_list
        self._image_processor = image_processor

    def set_preprocessing_flag(self, preprocessing_flag):
        self._reshape = preprocessing_flag

    def set_preprocessing_type(self,
                               preprocessing_type: str
                               ) -> None:
        """
        Set the desired pre-processing type. It must be between minmax and z-score.

        Args:
             preprocessing_type: the desired pre-processing.

        Returns:
            None
        """
        self._preprocessing_type = preprocessing_type

    def set_mean_std(self,
                     mean: torch.Tensor,
                     std: torch.Tensor
                     ) -> None:
        """
        Set custom values of mean and std for z-score normalization.

        Args:
             mean: torch.Tensor containing the desired mean along the three channels.
             std: torch.Tensor containing the desired standard deviation along the three channels.

        Returns:
            None
        """
        self._mean = mean
        self._std = std

    def _reset_mean_std(self) -> None:
        """
        Reset mean and std values to ImageNet ones.

        Returns:
            None
        """
        self._mean = [0.485, 0.456, 0.406]
        self._std = [0.229, 0.224, 0.225]
