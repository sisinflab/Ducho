from abc import abstractmethod
from src.internal.utils.human_sorting import human_sort
import os
import numpy


class DatasetFather:
    def __init__(self, input_directory_path, output_directory_path, model_name):
        self._framework_list = None
        self._model_name = model_name
        self._input_directory_path = input_directory_path
        self._output_directory_path = output_directory_path

        # the input path must already exist since is where are located the input file
        if not os.path.exists(self._input_directory_path):
            raise FileExistsError('input folder does not exists')
        # the output path can not exist but in this case it must be created
        if not os.path.exists(self._output_directory_path):
            os.makedirs(self._output_directory_path)

        # generate and order filenames
        # if the path is not a directory but a file, the filenames become the name of that single file
        if os.path.isfile(self._input_directory_path):
            self._filenames = ['']
            self._num_samples = 1
        else:
            self._filenames = os.listdir(self._input_directory_path)
            self._filenames = human_sort(self._filenames)
            self._num_samples = len(self._filenames)

    def __len__(self):
        return self._num_samples

    def set_model(self, model):
        self._model_name = model

    def create_output_file(self, index, extracted_data, model_layer):
        """

        Args:
            index: (int) is the index to the filenames list
            extracted_data: blob of data to put in the npy
            model_layer: the name of the layer

        Returns: it returns nothing to the program, but it creates a file as follows
                 datasetFolder/framework/modelName/modelLayer/fileName.npy

        """

        # generate file name
        input_file_name = self._filenames[index].split('.')[0]
        output_file_name = input_file_name + '.npy'

        # generate output path
        framework = self._framework_list[0]
        output_path = os.path.join(self._output_directory_path, framework)
        output_path = os.path.join(output_path, self._model_name)
        output_path = os.path.join(output_path, str(model_layer))
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # create file
        path = os.path.join(output_path, output_file_name)
        numpy.save(path, extracted_data)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def _pre_processing(self, sample):
        pass

    @abstractmethod
    def set_preprocessing_flag(self, preprocessing_flag):
        pass

    def set_framework(self, framework_list):
        self._framework_list = framework_list
