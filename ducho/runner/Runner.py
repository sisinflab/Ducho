import os
import logging
from tqdm import tqdm
from art import *
import torch
import tensorflow as tf
import datetime

from ducho.config.Config import Config
from ducho.multimodal.visual.VisualDataset import VisualDataset
from ducho.multimodal.textual.TextualDataset import TextualDataset
from ducho.multimodal.visual.VisualCnnFeatureExtractor import VisualCnnFeatureExtractor
from ducho.multimodal.textual.TextualCnnFeatureExtractor import TextualCnnFeatureExtractor
from ducho.multimodal.audio.AudioDataset import AudioDataset
from ducho.multimodal.audio.AudioCnnFeatureExtractor import AudioCnnFeatureExtractor


def _execute_extraction_from_models_list(models, extractor_class, gpu, dataset):
    """
    Takes in input the class of Dataset and Extractor, then for every model, for every layer of the model,
    Args:
        models: dicts of data (see Config.get_models)
        extractor_class: class Extractor
        gpu: gpu list
        dataset: class Dataset
    """

    change_mean_std = False

    for model in models:
        logging.info(f'Extraction model: {model["name"]}')

        extractor = extractor_class(gpu=gpu)

        # set framework
        extractor.set_framework(model['backend'])
        dataset.set_framework(model['backend'])

        # set model
        # extractor.set_model(model['name'])
        extractor.set_model(model)
        dataset.set_model(model['name'])

        # set preprocessing flag

        dataset.set_preprocessing_flag(model['preprocessing_flag'])
        if isinstance(dataset, VisualDataset):
            if 'preprceossing' in model.keys(): # preprocessing
                if not model['preprocessing'] in ['zscore', 'minmax', None]:
                    raise ValueError("Normalization must be 'minmax', 'zscore' or 'None'")
                dataset.set_preprocessing_type(model['preprocessing'])
            
            if 'mean' in model.keys() and 'std' in model.keys():
                if 'preprocessing' in model.keys(): 
                    if model['preprocessing'] == 'zscore':
                        dataset.set_mean_std(mean=model['mean'],
                                             std=model['std']
                                            )
                        change_mean_std = True
                    else:
                        raise ValueError("Mean and std values can be setted only for zscore normalization!") 
                else:
                    raise ValueError("Mean and std values can be setted only for zscore normalization but by the default normalization is None. Please, specify if you want zscore normalization!") 

        # execute extractions
        for model_layer in model['output_layers']:

            logging.info(f'Extraction layer: {model["name"]}.{model_layer}')

            # set output layer
            extractor.set_output_layer(model_layer)

            with tqdm(total=dataset.__len__()) as t:
                # for evey item do the extraction
                for index in range(dataset.__len__()):
                    # retrieve the item (preprocessed) from dataset
                    preprocessed_item, current_id = dataset.__getitem__(index)
                    # print(torch.max(preprocessed_item))
                    # print(torch.min(preprocessed_item))
                    # do the extraction
                    extractor_output = extractor.extract_feature(preprocessed_item)
                    # create the npy file with the extraction output
                    dataset.create_output_file((current_id if current_id else index), extractor_output, model_layer)
                    # update the progress bar
                    t.update()

            if change_mean_std:
                dataset._reset_mean_std()
                change_mean_std = False

            logging.info(f'Extraction with layer: {model["name"]}.{model_layer} is complete')

        logging.info(f'Extraction with model: {model["name"]} is complete')


class MultimodalFeatureExtractor:

    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        It instantiates the framework. Note the config file is a yml file
        Args:
             config_file_path: As a String, it could be the absolute path, or the path to the folder of the confg file
        """
        if not os.path.exists('./local/logs/'):
            os.makedirs('./local/logs/')

        log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt='%Y-%m-%d-%H:%M:%S',
            handlers=[
                logging.FileHandler(filename=f'./local/logs/{log_file}.log'),
                logging.StreamHandler()
            ]
        )

        framework = text2art("Ducho")
        logging.info('\n' + framework)
        logging.info('*** DUCHO: A Unified Framework for the Extraction of Multimodal Features in Recommendation ***')
        logging.info('*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***\n')
        self._config = Config(config_file_path, argv)
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()
        logging.info('Checking if CUDA version is compatible with TensorFlow and PyTorch...')
        logging.info(f'TENSORFLOW: Your tf version ({tf.__version__}) is compatible with you CUDA version!'
                     if len(tf.config.list_physical_devices("GPU")) > 0
                     else f'TENSORFLOW: Your tf version ({tf.__version__}) is not compatible with you CUDA version!')
        logging.info(f'PYTORCH: Your torch version ({torch.__version__}) is compatible with you CUDA version!'
                     if torch.cuda.is_available()
                     else f'TENSORFLOW: Your torch version ({torch.__version__}) is not compatible with you CUDA version!')

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
        """
        Executes only the item/visual extraction
        """
        if self._config.has_config('items', 'visual'):
            logging.info('Extraction on items for visual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'visual')
            models = self._config.get_models_list('items', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])

            logging.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=VisualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=visual_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_item_textual_extractions(self):
        """
        Executes only the item/textual extraction
        """
        if self._config.has_config('items', 'textual'):
            logging.info('Extraction on items for textual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'textual')
            models = self._config.get_models_list('items', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'],
                                             working_paths['output_path'],
                                             column=self._config.get_item_column())
            textual_dataset.set_type_of_extraction('items')

            logging.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=TextualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=textual_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_item_audio_extractions(self):
        """
        Executes only the item/audio extraction
        """
        if self._config.has_config('items', 'audio'):
            logging.info('Extraction on items for audio modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'audio')
            models = self._config.get_models_list('items', 'audio')
            # generate dataset and extractor
            audio_dataset = AudioDataset(working_paths['input_path'], working_paths['output_path'])

            logging.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=AudioCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=audio_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_visual_extractions(self):
        """
        Executes only the interaction/visual extraction
        """
        if self._config.has_config('interactions', 'visual'):
            logging.info('Extraction on interactions for visual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'visual')
            models = self._config.get_models_list('interactions', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])

            logging.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=VisualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=visual_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_textual_extractions(self):
        """
        Executes only the interaction/textual extraction
        """
        if self._config.has_config('interactions', 'textual'):
            logging.info('Extraction on interactions for textual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'textual')
            models = self._config.get_models_list('interactions', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'],
                                             working_paths['output_path'],
                                             column=self._config.get_interaction_column())

            logging.info('Extraction is starting...')
            textual_dataset.set_type_of_extraction('interactions')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=TextualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=textual_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_audio_extractions(self):
        """
        Executes only the interaction/audio extraction
        """
        if self._config.has_config('items', 'audio'):
            logging.info('Extraction on items for audio modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'audio')
            models = self._config.get_models_list('items', 'audio')
            # generate dataset and extractor
            audio_dataset = AudioDataset(working_paths['input_path'], working_paths['output_path'])

            logging.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=AudioCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=audio_dataset)
            logging.info(f'Extraction is complete, it\'s coffee break! ☕️')
