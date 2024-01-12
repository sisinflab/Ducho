import os
# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
#import logging  # Deprecated
from loguru import logger
from tqdm import tqdm
from alive_progress import alive_bar
#from art import *  # Deprecated
import torch
import tensorflow as tf
import datetime
import multiprocessing
from ducho.config.Config import Config
from ducho.multimodal.visual.VisualDataset import VisualDataset
from ducho.multimodal.textual.TextualDataset import TextualDataset
from ducho.multimodal.multiple.visual_textual.VisualTextualDataset import VisualTextualDataset
from ducho.multimodal.visual.VisualCnnFeatureExtractor import VisualCnnFeatureExtractor
from ducho.multimodal.textual.TextualCnnFeatureExtractor import TextualCnnFeatureExtractor
from ducho.multimodal.multiple.visual_textual.VisualTextualFeatureExtractor import VisualTextualFeatureExtractor
from ducho.multimodal.audio.AudioDataset import AudioDataset
from ducho.multimodal.audio.AudioCnnFeatureExtractor import AudioCnnFeatureExtractor
from ducho.internal.utils.json2dotnotation import banner


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
        logger.info(f'Extraction model: {model["model_name"]}')

        extractor = extractor_class(gpu=gpu)

        # set framework
        extractor.set_framework(model['backend'])
        dataset.set_framework(model['backend'])

        # set model
        # extractor.set_model(model['name'])
        extractor.set_model(model)
        dataset.set_model(model['model_name'])

        # set preprocessing flag

        dataset.set_preprocessing_flag(model['preprocessing_flag'])
        if isinstance(dataset, VisualDataset):
            if 'preprocessing' in model.keys(): # preprocessing
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
                        raise ValueError("Mean and std values can be set only for zscore normalization!")
                else:
                    raise ValueError("Mean and std values can be set only for zscore normalization but by the default normalization is None. Please, specify if you want zscore normalization!")

        # execute extractions
        for model_layer in model['output_layers']:

            logger.info(f'Extraction layer: {model["model_name"]}.{model_layer}')

            # set output layer
            extractor.set_output_layer(model_layer)

            if 'tensorflow' not in model['backend']:
                dataloader = torch.utils.data.DataLoader(dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         sampler=None,
                                                         num_workers=multiprocessing.cpu_count(),
                                                         pin_memory=True
                                                         )
                with alive_bar(len(dataloader)) as t:
                    # for evey item do the extraction
                    for index, batch in enumerate(dataloader):
                        # do the extraction
                        extractor_output = extractor.extract_feature(batch)
                        # create the npy file with the extraction output
                        dataset.create_output_file((index), extractor_output, model_layer)
                        # update the progress bar
                        t()
            else:
                with alive_bar(total=dataset.__len__()) as t:
                    # for evey item do the extraction
                    for index in range(dataset.__len__()):
                        # retrieve the item (preprocessed) from dataset
                        preprocessed_item = dataset.__getitem__(index)
                        # do the extraction
                        extractor_output = extractor.extract_feature(preprocessed_item)
                        # create the npy file with the extraction output
                        dataset.create_output_file((index), extractor_output, model_layer)
                        # update the progress bar
                        t()

            if change_mean_std:
                dataset._reset_mean_std()
                change_mean_std = False

            logger.success(f'Extraction with layer: {model["model_name"]}.{model_layer} is complete')

        logger.success(f'Extraction with model: {model["model_name"]} is complete')


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
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s [%(levelname)s] %(message)s",
        #     datefmt='%Y-%m-%d-%H:%M:%S',
        #     handlers=[
        #         logging.FileHandler(filename=f'./local/logs/{log_file}.log'),
        #         logging.StreamHandler()
        #     ]
        # )

        logger.add(f"./local/logs/{log_file}.log", level="INFO",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
        #logging.add(lambda msg: print(msg, end=""), level="INFO")  # Stream handler for console output


        logger.info('\n' + banner)
        logger.log("WELCOME",'*** DUCHO: A Unified Framework for the Extraction of Multimodal Features in Recommendation ***')
        logger.log("WELCOME",'*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***\n')
        self._config = Config(config_file_path, argv)
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()
        logger.info('Checking if CUDA version is compatible with TensorFlow and PyTorch...')
        if len(tf.config.list_physical_devices("GPU")) > 0:
            logger.info(f'TENSORFLOW: Your tf version ({tf.__version__}) is compatible with your CUDA version!')
        else:
            logger.error(f'TENSORFLOW: Your tf version ({tf.__version__}) is not compatible with your CUDA version!')
        if torch.cuda.is_available():
            logger.info(f'PYTORCH: Your torch version ({torch.__version__}) is compatible with your CUDA version!')
        else:
            logger.error(f'PYTORCH: Your torch version ({torch.__version__}) is not compatible with your CUDA version!')

    def execute_extractions(self):
        """
        It executes all the extraction that have to be done
        """
        self.do_item_visual_extractions()
        self.do_interaction_visual_extractions()
        self.do_item_textual_extractions()
        self.do_interaction_textual_extractions()
        self.do_interaction_audio_extractions()
        self.do_item_visual_textual_extractions()

    def do_item_visual_textual_extractions(self):
        """
        Executes only the visual-textual items extraction
        """
        if self._config.has_config('items', 'visual_textual'):
            logger.info('Extraction on items for visual_textual modality')

            # get paths and models
            working_paths = self._config.paths_for_multiple_extraction('items', 'visual_textual')
            models = self._config.get_models_list('items', 'visual_textual')
            # generate dataset and extractor
            visual_textual_dataset = VisualTextualDataset(working_paths['input_path'],
                                                          working_paths['output_path'],
                                                          column=self._config.get_item_column())
            visual_textual_dataset._textual_dataset.set_type_of_extraction('items')

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=VisualTextualFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=visual_textual_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_item_visual_extractions(self):
        """
        Executes only the item/visual extraction
        """
        if self._config.has_config('items', 'visual'):
            logger.info('Extraction on items for visual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'visual')
            models = self._config.get_models_list('items', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=VisualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=visual_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_item_textual_extractions(self):
        """
        Executes only the item/textual extraction
        """
        if self._config.has_config('items', 'textual'):
            logger.info('Extraction on items for textual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'textual')
            models = self._config.get_models_list('items', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'],
                                             working_paths['output_path'],
                                             column=self._config.get_item_column())
            textual_dataset.set_type_of_extraction('items')

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=TextualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=textual_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_item_audio_extractions(self):
        """
        Executes only the item/audio extraction
        """
        if self._config.has_config('items', 'audio'):
            logger.info('Extraction on items for audio modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'audio')
            models = self._config.get_models_list('items', 'audio')
            # generate dataset and extractor
            audio_dataset = AudioDataset(working_paths['input_path'], working_paths['output_path'])

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=AudioCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=audio_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_visual_extractions(self):
        """
        Executes only the interaction/visual extraction
        """
        if self._config.has_config('interactions', 'visual'):
            logger.info('Extraction on interactions for visual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'visual')
            models = self._config.get_models_list('interactions', 'visual')
            # generate dataset and extractor
            visual_dataset = VisualDataset(working_paths['input_path'], working_paths['output_path'])

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=VisualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=visual_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_textual_extractions(self):
        """
        Executes only the interaction/textual extraction
        """
        if self._config.has_config('interactions', 'textual'):
            logger.info('Extraction on interactions for textual modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('interactions', 'textual')
            models = self._config.get_models_list('interactions', 'textual')
            # generate dataset and extractor
            textual_dataset = TextualDataset(working_paths['input_path'],
                                             working_paths['output_path'],
                                             column=self._config.get_interaction_column())

            logger.info('Extraction is starting...')
            textual_dataset.set_type_of_extraction('interactions')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=TextualCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=textual_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')

    def do_interaction_audio_extractions(self):
        """
        Executes only the interaction/audio extraction
        """
        if self._config.has_config('items', 'audio'):
            logger.info('Extraction on items for audio modality')

            # get paths and models
            working_paths = self._config.paths_for_extraction('items', 'audio')
            models = self._config.get_models_list('items', 'audio')
            # generate dataset and extractor
            audio_dataset = AudioDataset(working_paths['input_path'], working_paths['output_path'])

            logger.info('Extraction is starting...')
            _execute_extraction_from_models_list(models=models,
                                                 extractor_class=AudioCnnFeatureExtractor,
                                                 gpu=self._config.get_gpu(),
                                                 dataset=audio_dataset)
            logger.success(f'Extraction is complete, it\'s coffee break! ☕️')
