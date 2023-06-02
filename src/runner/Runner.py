import os
import logging
from tqdm import tqdm
from art import *
import tensorflow as tf
import torch

from src.config.Config import Config
from src.multimodal.visual.VisualDataset import VisualDataset
from src.multimodal.textual.TextualDataset import TextualDataset
from src.multimodal.visual.VisualCnnFeatureExtractor import VisualCnnFeatureExtractor
from src.multimodal.textual.TextualCnnFeatureExtractor import TextualCnnFeatureExtractor
from src.multimodal.audio.AudioDataset import AudioDataset
from src.multimodal.audio.AudioCnnFeatureExtractor import AudioCnnFeatureExtractor


def _execute_extraction_from_models_list(models, extractor, dataset, modality_type):
    """
    Takes in input the class of Dataset and Extractor, then for every model, for every layer of the model,
    :param models: dicts of data (see Config.get_models)
    :param extractor: class Extractor
    :param dataset: class Dataset
    :param modality_type: 'audio'/'visual'/'textual'
    """
    for model in models:
        logging.info(' Now using model: %s', str(model['name']))

        # set framework
        extractor.set_framework(model['framework'])
        dataset.set_framework(model['framework'])

        # set model
        # extractor.set_model(model['name'])
        extractor.set_model(model)
        dataset.set_model(model['name'])

        # set preprocessing flag
        dataset.set_preprocessing_flag(model['preprocessing_flag'])

        # execute extractions
        for model_layer in model['output_layers']:

            logging.info(' Now using layer: - %s', str(model_layer))

            # set output layer
            extractor.set_output_layer(model_layer)

            with tqdm(total=dataset.__len__()) as t:
                # for evey item do the extraction
                for index in range(dataset.__len__()):
                    # retrieve the item (preprocessed) from dataset
                    preprocessed_item = dataset.__getitem__(index)
                    # do the extraction
                    extractor_output = extractor.extract_feature(preprocessed_item)
                    # create the npy file with the extraction output
                    dataset.create_output_file(index, extractor_output, model_layer)
                    # update the progress bar
                    t.update()


class MultimodalFeatureExtractor:

    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        It instantiates the framework. Note the config file is a yml file
        :param config_file_path: As a String, it could be the absolute path, or the path to the folder of the confg file
        """
        framework = text2art("Ducho")
        print(framework)
        print('*** A Unified Framework for the Extraction of Multimodal Features in Recommendation ***')
        print('*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***')
        print()
        self._config = Config(config_file_path, argv)
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()
        logging.info('Checking if CUDA version is compatible with TensorFlow and PyTorch...')
        logging.info(f'TENSORFLOW: {tf.config.list_physical_devices("GPU")}')
        logging.info(f'PYTORCH: {torch.cuda.is_available()}')

    def execute_extractions(self):
        """
        It executes all the extraction that have to be done
        """
        self.do_item_visual_extractions()
        self.do_interaction_visual_extractions()
        self.do_item_textual_extractions()
        self.do_interaction_textual_extractions()
        self.do_interaction_audio_extractions()

    def do_item_visual_extractions(self):
        if self._config.has_config('items', 'visual'):
            logging.info(' Config for visual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'visual')
            models = self._config.get_models_list('items', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = VisualCnnFeatureExtractor(self._config.get_gpu())

            # visual_dataset.set_model_map(self._config.get_model_map_path())
            # cnn_feature_extractor.set_model_map(self._config.get_model_map_path())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))
            _execute_extraction_from_models_list(models, cnn_feature_extractor, visual_dataset, 'visual')

    def do_item_textual_extractions(self):
        if self._config.has_config('items', 'textual'):
            logging.info(' Config for textual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'textual')
            models = self._config.get_models_list('items', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = TextualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))

            textual_dataset.set_type_of_extraction('items')
            _execute_extraction_from_models_list(models, cnn_feature_extractor, textual_dataset, 'textual')

    def do_interaction_visual_extractions(self):
        if self._config.has_config('interactions', 'visual'):
            logging.info(
                ' Config for visual extractions from interactions detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'visual')
            models = self._config.get_models_list('interactions', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = VisualCnnFeatureExtractor(self._config.get_gpu())

            # visual_dataset.set_model_map(self._config.get_model_map_path())
            # cnn_feature_extractor.set_model_map(self._config.get_model_map_path())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))
            _execute_extraction_from_models_list(models, cnn_feature_extractor, visual_dataset, 'visual')

    def do_interaction_textual_extractions(self):
        if self._config.has_config('interactions', 'textual'):
            logging.info(' Config for textual extractions from items detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'textual')
            models = self._config.get_models_list('interactions', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = TextualCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))

            textual_dataset.set_type_of_extraction('interactions')
            _execute_extraction_from_models_list(models, cnn_feature_extractor, textual_dataset, 'textual')

    # DEPRECATED:
    def __execute_extractions_second_delete(self):
        visual_work_env_ls = self._config.get_visual_working_environment_list()
        print(visual_work_env_ls)
        # self._execute_visual_extractions(visual_work_env_ls)
        textual_work_env_ls = self._config.get_textual_working_environment_list()
        print(textual_work_env_ls)

    # DEPRECATED
    def __execute_visual_extractions_delete(self, work_env_ls):
        for work_env in work_env_ls:
            if self._config.has_config(work_env, 'visual'):
                # logging...
                # get paths and models
                working_paths = self._config.paths_for_extraction(work_env, 'visual')
                models = self._config.get_models_list(work_env, 'visual')
                # generate dataset and extractor
                visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])
                cnn_feature_extractor = VisualCnnFeatureExtractor(self._config.get_gpu())

                # logging.info(' Working environment created')
                # logging.info(' Number of models to use: %s', str(models.__len__()))
                _execute_extraction_from_models_list(models, cnn_feature_extractor, visual_dataset, 'visual')

    def do_interaction_audio_extractions(self):
        if self._config.has_config('interactions', 'audio'):
            logging.info(
                ' Config for audio extractions from interactions detected, the extraction is going to start ...')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'audio')
            models = self._config.get_models_list('interactions', 'audio')
            # generate dataset and extractor
            audio_dataset = AudioDataset(working_paths['input_path'], working_paths['output_path'])
            cnn_feature_extractor = AudioCnnFeatureExtractor(self._config.get_gpu())

            logging.info(' Working environment created')
            logging.info(' Number of models to use: %s', str(models.__len__()))

            _execute_extraction_from_models_list(models, cnn_feature_extractor, audio_dataset, 'audio')
