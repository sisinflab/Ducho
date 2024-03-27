from abc import abstractmethod
from ducho.internal.utils.human_sorting import human_sort
import os
import numpy
from loguru import logger


class DatasetFather:
    """
    Represents a dataset handler object.

    This class provides functionality to manage dataset directories and filenames for data extraction.

    Attributes:
        _backend_libraries_list: A list of backend libraries (e.g. Tensorflow, Pytorch, Transformers)
        _model_name (str): The name of the model.
        _input_directory_path (str or dict): The path to the input directory or a dictionary containing data paths.
        _output_directory_path (str): The path to the output directory, where the extraction will be saved.
    """
    def __init__(self, input_directory_path, output_directory_path, model_name):
        """
        Initialize the DatasetFather object.

        Args:
            input_directory_path (str or dict): The path to the input directory or a dictionary containing paths.
            output_directory_path (str): The path to the output directory.
            model_name (str): The name of the model.

        Returns:
            None
        """
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
        if type(self._input_directory_path) == dict:
            self._filenames, self._num_samples = dict(), dict()
            for k, v in self._input_directory_path.items():
                if os.path.isfile(v):
                    self._filenames[k] = ['']
                    self._num_samples[k] = 1
                else:
                    current_filenames = os.listdir(v)
                    self._filenames[k] = human_sort(current_filenames)
                    self._num_samples[k] = len(self._filenames[k])
        else:
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

    def create_output_file(self, input_batch, extracted_data, model_layer, fusion=None):
        """
        Create an output numpy file with extracted data.
        (E.g. datasetFolder/framework/modelName/modelLayer/fileName.npy)

        Args:
            input_batch (tensor): The batch just processed by the extractor. It contains the filenames too.
            extracted_data (Any): The data to be stored in the .npy file.
            model_layer (str): The name of the layer.
            fusion (str, optional): The type of fusion for multimodal models.

        Returns:
            None

        """

        backend_library = self._backend_libraries_list[0]
        output_path = os.path.join(self._output_directory_path, backend_library)
        output_path = os.path.join(output_path, os.path.splitext(os.path.basename(self._model_name))[0])
        output_path = os.path.join(output_path, str(model_layer))

        filenames = input_batch[1]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # checking whether batch size is > 1.

        if len(extracted_data) > 1:
            filenames = [f.split('.')[0] for f in filenames]
            for f, e in zip(filenames, extracted_data):
                output_file_name = f + '.npy'
                path = os.path.join(output_path, output_file_name)
                numpy.save(path, e)

        else:
            filenames = filenames.split('.')[0]
            output_file_name = filenames + '.npy'
            path = os.path.join(output_path, output_file_name)
            numpy.save(path, extracted_data)
    # def create_output_file(self, index, extracted_data, model_layer, fusion=None):
    #     """
    #     Create an output numpy file with extracted data.
    #     (E.g. datasetFolder/framework/modelName/modelLayer/fileName.npy)

    #     Args:
    #         index (int): The index to the filenames list.
    #         extracted_data (Any): The data to be stored in the .npy file.
    #         model_layer (str): The name of the layer.
    #         fusion (str, optional): The type of fusion for multimodal models.

    #     Returns:
    #         None

    #     """

    #     # Generate file name
    #     input_file_name = self._filenames[index].split('.')[0]
    #     output_file_name = input_file_name + '.npy'

    #     # Generate output path
    #     backend_library = self._backend_libraries_list[0]
    #     output_path = os.path.join(self._output_directory_path, backend_library)
    #     output_path = os.path.join(output_path, os.path.splitext(os.path.basename(self._model_name))[0])
    #     output_path = os.path.join(output_path, str(model_layer))
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)

    #     # Create file
    #     path = os.path.join(output_path, output_file_name)
    #     numpy.save(path, extracted_data)

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
        Set the framework(s) to use.

        Args:
            backend_libraries_list (list of str): A list of strings representing the framework(s) to use.
                It's acceptable to have only one item in the list.

        Returns:
            None

        """
        self._backend_libraries_list = backend_libraries_list
