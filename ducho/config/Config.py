import os
from loguru import logger
import json
from ducho.internal.utils.YamlFileManager import YamlFileManager
from ducho.internal.utils.json2dotnotation import parse_and_print
import warnings
import copy


def _clean_preprocessing_flag_of_models(model, type_of_extraction):
    """
    Clean preprocessing flags of models by renaming them under the same name for future data management.

    Args:
        model: The model object.
        type_of_extraction: The type of data extraction: Textual, Visual, and Audio

    Returns:
        dict: The model object with the renamed preprocessing flag.
    """
    data_flag = ''

    if type_of_extraction == 'textual':
        data_flag = model.pop('clear_text') if 'clear_text' in model.keys() else False
    elif type_of_extraction == 'visual':
        if 'reshape' in model.keys():
            data_flag = model.pop('reshape')
            if 'transformers' in model['backend']:
                logger.warning(f"Custom reshape may be overridden by predefined HuggingFace transformers preprocessing module's configurations")

    elif type_of_extraction == 'audio':
        # Right now there is no preprocessing flag but one is needed for code clearance
        data_flag = None

    model.update({'preprocessing_flag': data_flag})
    return model


def _clean_unique_flags_of_models(model, type_of_extraction):
    if type_of_extraction == 'textual':
        # to maintain the runner agnostic, when it gives the model name to the extractor, it also need to give it the
        # task that the model have to do.
        # so in textual...
        print('nah, after')


class Config:
    """
    Manage the configuration within the config YAML file.

    These configurations are needed to define what extracions to perform.

    """
    def __init__(self, config_file_path, argv):
        """
        Initialize ConfigurationManager with the specified configuration file path and command-line arguments.

        Args:
            config_file_path (str): Path to the config YAML file.
            argv (list): Runner's arguments.

        Returns:
            None
        """
        # both absolute and relative path are fine
        self._yaml_manager = YamlFileManager(config_file_path)

        self._data_dict = self._yaml_manager.get_default_dict()
        custom_data_dict = self._yaml_manager.get_raw_dict()
        self._data_dict.update(custom_data_dict)

        if config_file_path != './config/config.yml':
            logger.warning(f'Custom configuration file {config_file_path} provided. Will override the default one')
        else:
            logger.info('No custom configuration file provided. Will use the default one')

        if argv:
            for kv in argv:
                self.__update_dict(kv.split("=")[0], kv.split("=")[1])

        self._data_dict = self.__clean_dict(self._data_dict)

        logger.info(f'Loaded configuration:\n\n{parse_and_print(self._data_dict)}\n')

    def __update_dict(self, keys_as_string, value):
        """
        Update the configuration dictionary with a new key-value pair, where keys are provided as a string.

        Args:
            keys_as_string (str): The keys as a string separated by '--' and '.'.
            value: The value to set for the specified keys.
        Returns:
            None
        """
        def sub_of_update_dict(lists_of_keys, last_value, sub_dict):
            """
            Recursively update a sub-dictionary with the given keys and value.

            Args:
                lists_of_keys (list): List of keys.
                last_value: The value to set.
                sub_dict (dict): The sub-dictionary to update.

            Returns:
                dict: The updated sub-dictionary.
            """
            if len(lists_of_keys) == 1:
                sub_dict.update({lists_of_keys.pop(0): last_value})
                return sub_dict
            else:
                key_to_use = lists_of_keys.pop(0)
                sub_dict.update({key_to_use: sub_of_update_dict(ls_of_keys, last_value, sub_dict[key_to_use])})
                return sub_dict

        ls_of_keys = keys_as_string.split('--')[1].split('.')
        first_key = ls_of_keys.pop(0)
        if len(ls_of_keys) > 0:
            self._data_dict.update({first_key: sub_of_update_dict(ls_of_keys, value, self._data_dict[first_key])})
        else:
            self._data_dict.update({first_key: value})

    def __clean_dict(self, data):
        """
        It cleans the dict to be easily read in the future.
        It crosses in every element of the dict in search of a list of dict to transform in a big dict:
        if there is a dict, it crosses every value (recalling this method).
        If there is a list, it crosses every item (recalling this method). then if the items are dicts the list
        is swapped with a big dict
        Args:
            data: it's the data contained in the yaml file as a dict

        Returns:
            data: it returns data cleaned, every list of dict is transformed in a single dict

        """
        # using yaml there is a problem:
        # it has no strict rules, so you can have [[{}]] [[]] {[]} {{}} ecc
        # this recursive method transform everything as {...{}...} or {...[]...}
        temp_dict = {}
        if isinstance(data, dict):
            for key in data.keys():
                # the model dict follow a particular configuration that is necessary not to change
                if key != 'model':
                    value = self.__clean_dict(data[key])
                    data.update({key: value})
        if isinstance(data, list):
            for element in data:
                element = self.__clean_dict(element)
                # the following code follow a statement that is always true using yaml:
                # if in the list one element is a dict, so are all the others elements
                if isinstance(element, dict):
                    temp_dict.update(element)
        if bool(temp_dict):
            data = temp_dict
        return data

    def get_gpu(self):
        """
        Get the GPU list as a string.

        Returns:
            str: The GPU list as a string.
        """
        # if there is not a gpu config then "-1" (use cpu only)
        # otherwise return the config
        if 'gpu list' in self._data_dict:
            gpu_list = self._data_dict['gpu list']
            if isinstance(gpu_list, str):
                # es '1' or '1,2'
                return gpu_list
            elif isinstance(gpu_list, int):
                # es 1 -> '1'
                return str(gpu_list)
            elif isinstance(gpu_list, list):
                # es [1,3] -> '1,3'
                return ','.join(str(x) for x in gpu_list)
            else:
                raise SyntaxError('the gpu list is written in a incorrect way')
        else:
            return '-1'

    def get_extractions(self):
        """
        Get the extraction configurations.

        Returns:
            dict: A dictionary containing extraction configurations for visual, textual, and visual_textual data.
        """
        extractions_dict = {key: copy.deepcopy(self._data_dict[key]) for key in ['visual', 'textual', 'visual_textual'] if key in self._data_dict}
        return extractions_dict

    def has_config(self, origin_of_elaboration, type_of_extraction):
        """
        Check if the configuration contains values for the specified origin of elaboration and type of extraction.

        Args:
            origin_of_elaboration (str): Either 'items' or 'interactions'.
            type_of_extraction (str): Either 'textual', 'visual', or 'audio'.

        Returns:
            bool: True if the configuration contains values, False otherwise.
        """
        if type_of_extraction in self._data_dict and origin_of_elaboration in self._data_dict[type_of_extraction]:
            local_dict = self._data_dict[type_of_extraction][origin_of_elaboration]
            # check if local dict has input/output/model
            if 'input_path' in local_dict and 'output_path' in local_dict and 'model' in local_dict:
                # in this case it's all right but must be checked that the values are not empty
                input_value = local_dict['input_path']
                output_value = local_dict['output_path']
                model_value = local_dict['model']
                if input_value is not None and output_value is not None and model_value is not None:
                    return True
        return False

    def get_columns(self, modality):
        """
        Gives the column names in the case of textual and visual_textual modalities
        Args:
            modality: 'textual', 'visual', 'audio', or 'visual_textual'

        Returns:
            for items: two dicts as { 'item_column': column name for items' ids, 'text_column': column name for items' descriptions }
            for interactions: two dicts as { 'interaction_columns': column name for items' ids, 'text_column': column name for items' descriptions }
        """
        if 'items' in self._data_dict[modality]:
            if 'item_column' in self._data_dict[modality]['items'] and 'text_column' in self._data_dict[modality]['items']:
                return self._data_dict[modality]['items']['item_column'], self._data_dict[modality]['items']['text_column']
            else:
                raise KeyError('Please, specify both item_column and text_column in the configuration file!')
        elif 'interactions' in self._data_dict[modality]:
            if 'interaction_columns' in self._data_dict[modality]['interactions'] and 'text_column' in self._data_dict[modality]['interactions']:
                if len(self._data_dict[modality]['interactions']['interaction_columns']) != 2:
                    raise ValueError('The value of interaction_columns in the configuration file should be a 2-length list!')
                return self._data_dict[modality]['interactions']['interaction_columns'], self._data_dict[modality]['interactions']['text_column']
            else:
                raise KeyError('Please, specify both interaction_columns and text_column in the configuration file!')

    def paths_for_extraction(self, origin_of_elaboration, type_of_extraction):
        """
        Get the working environments for extraction.

        Args:
            origin_of_elaboration (str): Either 'items' or 'interactions'.
            type_of_extraction (str): Either 'textual', 'visual', or 'audio'.

        Returns:
            dict: A dictionary containing input and output paths.
        """
        # {'input_path': ///, 'output_path': ///}
        relative_input_path = self._data_dict[type_of_extraction][origin_of_elaboration]['input_path']
        relative_output_path = self._data_dict[type_of_extraction][origin_of_elaboration]['output_path']

        return {
            'input_path': os.path.join(self._data_dict['dataset_path'], relative_input_path),
            'output_path': os.path.join(self._data_dict['dataset_path'], relative_output_path)}

    def paths_for_multiple_extraction(self, origin_of_elaboration, type_of_extraction):
        """
        Get the working environments for multiple extraction.

        Args:
            origin_of_elaboration (str): Either 'items' or 'interactions'.
            type_of_extraction (str): Either 'textual', 'visual', or 'audio'.

        Returns:
            dict: A dictionary containing input and output paths for multiple extractions.
        """
        # {'input_path': ///, 'output_path': ///}
        relative_input_path = self._data_dict[type_of_extraction][origin_of_elaboration]['input_path']
        relative_output_path = self._data_dict[type_of_extraction][origin_of_elaboration]['output_path']

        return {
            'input_path': {k: os.path.join(self._data_dict['dataset_path'], relative_input_path[k])
                            for k, _ in relative_input_path.items()},
            'output_path': {k: os.path.join(self._data_dict['dataset_path'], relative_output_path[k])
                            for k, _ in relative_input_path.items()}}

    def get_models_list(self, origin_of_elaboration, type_of_extraction):
        """
        Get the list of models.

        Args:
            origin_of_elaboration (str): Either 'items' or 'interactions'.
            type_of_extraction (str): Either 'textual', 'visual', or 'audio'.

        Returns:
            list: A list of model dictionaries, where each model contains the following keys:
                - 'name': The name of the model.
                - 'output_layers': The layers of extraction.
                - 'framework': The framework to work with (e.g., 'tensorflow', 'torch', 'transformers').
                - 'preprocessing_flag': A custom flag to manage the preprocessing of the data.
        """
        models = self._data_dict[type_of_extraction][origin_of_elaboration]['model']

        for model in models:

            if 'torch' in model['backend'] and isinstance(model['output_layers'], int):
                raise ValueError('Please specify the output layer name and not its index.')

            # output_layers has to be a list
            if not isinstance(model['output_layers'], list):
                model.update({'output_layers': [model['output_layers']]})

            # preprocessing flags cleaning
            model = _clean_preprocessing_flag_of_models(model, type_of_extraction)

            # Framework elaboration
            # - if INPUT FRAMEWORK is ['tensorflow', 'torch'] then two different model dicts will be added to the list,
            #   each one identical to the other except for the fact that it contains only one of the 2 type of framework
            #   WARNING: the feature to do both of them in the same model declaration is forbidden since they use
            #   different way to call their layers
            # - if OUTPUT FRAMEWORK is ['tensorflow', 'torch'] then outside of this method it means that
            #   the framework in which operate is not known but only one of them will be executed

            # this name refers to the way the libraries of models are called in Ducho
            library_key = 'backend'
            if library_key in model.keys():
                framework_value = model[library_key]

                if framework_value == ['tensorflow', 'torch']:
                    # this setting does not work properly because the two framework used calls different layers
                    first_model = model
                    first_model.update({library_key: ['tensorflow']})

                    second_model = model
                    second_model.update({library_key: ['torch']})

                    # layers

                    first_model_layers = []
                    second_model_layers = []
                    for layer in model['output_layers']:
                        if isinstance(layer, int):
                            second_model_layers.append(layer)
                        else:
                            first_model_layers.append(layer)

                    first_model.update({'output_layers': first_model_layers})
                    second_model.update({'output_layers': second_model_layers})

                    # models_list.append(second_model)
                    # models_list.append(first_model)

                    # this setting does not work properly because the two framework used calls different layers
                    raise ValueError(' unfortunately calling both framework simultaneity doesnt work')
                # framework value must be a list
                elif isinstance(framework_value, str):
                    model.update({library_key: [framework_value]})

                # the following elif was written with the idea that every type of extraction would have only torch or
                # tensorflow. Now this only make sense in the visual case
                #   elif framework_value != ['tensorflow'] and framework_value != ['torch']:
                #       raise ValueError('the framework tag in the yaml file is not written correctly')
            else:
                # the framework is not set
                if type_of_extraction == 'textual':
                    # textual case
                    # in this case we use the 'transformers' framework
                    model.update({library_key: ['transformers', 'sentence-transformers']})
                elif type_of_extraction == 'visual':
                    # it is in the visual case, it uses tensorflow or torch, but doesn't know which one
                    # so both are set as plausible
                    model.update({library_key: ['tensorflow', 'torch']})
                elif type_of_extraction == 'audio':
                    # it is the audio case, it uses torchaudio or transformers
                    # both are plausible, it will try torchaudio and if the model is not in its list, it will try
                    # transformers
                    model.update({library_key: ['torch', 'transformers']})

        return models


# Configure custom loguru levels
logger.configure(
    levels=[dict(name="NEW", no=13, icon="¤", color=""), dict(name="WELCOME", no=25, color="<green>", icon="!!!")],
)

# Hide Torch warnings
warnings.filterwarnings("ignore")
