import torch
from loguru import logger

from ducho.multimodal.visual.VisualDataset import VisualDataset
from ducho.multimodal.textual.TextualDataset import TextualDataset
import numpy, os


class VisualTextualDataset:
    """
    This class represents the Visual-Textual Dataset used for the data loading process.
    """
    def __init__(self,
                 input_directory_path,
                 output_directory_path,
                 columns=None,
                 model_name='openai/clip-vit-base-patch32',
                 reshape=(224, 224)):
        """
        It manages the Visual-Textual Dataset, which consists of a folder containing input data and another folder for output data.
        It handles the preprocessing of input data and manages the output data.

        Args:
            input_directory_path: A string representing the path to the folder containing the input data to be processed.
            output_directory_path: A string representing the path to the folder where the output data will be stored. If the folder does not exist, it will be created.
            columns: A list of columns to be considered.
            model_name: A string specifying the model to be used. This can be reset later.
            reshape: A tuple (int, int) representing the width and height for resizing the input images. This can be reset later.

        Returns:
            None
        """
        self._backend_libraries_list = None
        self._model_name = model_name
        self._reshape = reshape
        self.input_image_path, self.input_text_path = input_directory_path['visual'], input_directory_path['textual']
        self.output_image_path, self.output_text_path = output_directory_path['visual'], output_directory_path['textual']
        self._visual_dataset = VisualDataset(self.input_image_path, self.output_image_path, model_name)
        self._textual_dataset = TextualDataset(self.input_text_path, self.output_text_path, columns)
        self.set_framework = self._visual_dataset.set_framework
        self.set_model = self._visual_dataset.set_model
        self.set_preprocessing_flag = self._visual_dataset.set_preprocessing_flag

    def __len__(self):
        return min(self._visual_dataset._num_samples, self._textual_dataset._num_samples)

    def __getitem__(self, idx):
        visual_input = self._visual_dataset.__getitem__(idx)
        textual_input, _ = self._textual_dataset.__getitem__(idx)
        return visual_input, textual_input

    def create_output_file(self, index, extracted_data, model_layer, fusion=None):
        """
        This procedure is responsible for generating output files.

        Args:
            index: The index of the file to be processed.
            extracted_data: A tuple containing the extracted features.
            model_layer: The name of the output layer for the selected model.
            fusion: A string indicating the type of fusion to perform. If None, the procedure generates two separate output files.
                    Otherwise, it creates a single output file based on the specified fusion type.


        Returns:
            None
        """
        # generate file name
        input_file_name = self._visual_dataset._filenames[index].split('.')[0]
        output_file_name = input_file_name + '.npy'

        # generate output path
        backend_library = self._visual_dataset._backend_libraries_list[0]

        if not fusion:
            # visual
            output_image_path = os.path.join(self.output_image_path, backend_library)
            output_image_path = os.path.join(output_image_path, self._model_name)
            output_image_path = os.path.join(output_image_path, str(model_layer))
            if not os.path.exists(output_image_path):
                os.makedirs(output_image_path)
            # create file
            path = os.path.join(output_image_path, output_file_name)
            numpy.save(path, extracted_data[0])

            # textual
            output_text_path = os.path.join(self.output_text_path, backend_library)
            output_text_path = os.path.join(output_text_path, self._model_name)
            output_text_path = os.path.join(output_text_path, str(model_layer))
            if not os.path.exists(output_text_path):
                os.makedirs(output_text_path)
            # create file
            path = os.path.join(output_text_path, output_file_name)
            numpy.save(path, extracted_data[1])
        else:
            last_image_path = os.path.basename(os.path.normpath(self.output_image_path))
            last_text_path = os.path.basename(os.path.normpath(self.output_text_path))
            first_path = self.output_image_path.replace(last_image_path, '')
            output_path = f'{last_image_path}_{last_text_path}_{fusion}'
            output_path = os.path.join(first_path, output_path, backend_library)
            output_path = os.path.join(output_path, self._model_name)
            output_path = os.path.join(output_path, str(model_layer))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            # create file
            path = os.path.join(output_path, output_file_name)
            # fusion
            if fusion == 'concat':
                extracted_data = numpy.concatenate(extracted_data, axis=1)
            elif fusion == 'sum':
                if extracted_data[0].shape != extracted_data[1].shape:
                    raise ValueError(f'The shapes of visual and textual embeddings should be the same for {fusion} fusion!')
                extracted_data = numpy.add(extracted_data[0], extracted_data[1])
            elif fusion == 'mul':
                if extracted_data[0].shape != extracted_data[1].shape:
                    raise ValueError(f'The shapes of visual and textual embeddings should be the same for {fusion} fusion!')
                extracted_data = numpy.multiply(extracted_data[0], extracted_data[1])
            elif fusion == 'mean':
                if extracted_data[0].shape != extracted_data[1].shape:
                    raise ValueError(f'The shapes of visual and textual embeddings should be the same for {fusion} fusion!')
                extracted_data = numpy.mean(extracted_data, axis=0)

            numpy.save(path, extracted_data)

    def set_reshape(self, reshape):
        """
        Set the reshape variable according to the desired value.

        Args:
             reshape: Tuple (int, int) representing the width and height for resizing the input.

        Returns:
            None
        """
        self._reshape = reshape

    def set_model_name(self, model_name):
        """
        Set the model name for the serialization dir.

        Args:
             model_name: name of the multimodal model

        Returns:
            None
        """
        self._model_name = model_name