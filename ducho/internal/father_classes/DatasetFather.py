from abc import abstractmethod
from ducho.internal.utils.human_sorting import human_sort
import os
import numpy
from loguru import logger


class DatasetFather:
    def __init__(self, input_directory_path, output_directory_path, model_name):
        self._backend_libraries_list = None
        self._model_name = model_name
        self._input_directory_path = input_directory_path
        self._output_directory_path = output_directory_path

        # the input path must already exist since is where are located the input file
        if not os.path.exists(self._input_directory_path):
            raise FileExistsError('input folder does not exists')

        logger.info(f'Reading files from: {os.path.abspath(self._input_directory_path)}')
        # the output path can not exist but in this case it must be created
        if not os.path.exists(self._output_directory_path):
            logger.info(f'Output directory does not exist. Will create it in: {os.path.abspath(self._output_directory_path)}')
            os.makedirs(self._output_directory_path)
        else:
            logger.warning('The output directory already exists. This extraction could overwrite existing files!')

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

        Returns:
            it returns nothing to the program, but it creates a file as follows :
            datasetFolder/framework/modelName/modelLayer/fileName.npy

        """

        # generate file name
        input_file_name = self._filenames[index].split('.')[0]
        output_file_name = input_file_name + '.npy'

        # generate output path
        backend_library = self._backend_libraries_list[0]
        output_path = os.path.join(self._output_directory_path, backend_library)
        output_path = os.path.join(output_path, os.path.splitext(os.path.basename(self._model_name))[0])
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

    def set_framework(self, backend_libraries_list):
        """
        It set the framework to use as e.g: 'torch', 'tensorflow', 'transformers', 'torchaudio'

        Args:
            backend_libraries_list: the list of String of the framework. It's acceptable to have only one item in the list

        """
        self._backend_libraries_list = backend_libraries_list
