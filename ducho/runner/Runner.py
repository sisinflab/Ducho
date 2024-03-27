import os
from loguru import logger
from alive_progress import alive_bar
import torch
import platform
import tensorflow as tf
import importlib
import datetime
import multiprocessing
from ducho.config.Config import Config
from ducho.multimodal.visual.VisualDataset import VisualDataset
from ducho.multimodal.multiple.visual_textual.VisualTextualDataset import VisualTextualDataset
from ducho.internal.utils.json2dotnotation import banner

# Hiding TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
# Hiding Transformers' tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def camel_case(s):
    return ''.join(x for x in s.title() if not x.isspace()).replace('_', '')


def _execute_extraction_from_models_list(models, extractor_class, gpu, dataset):
    """
    Execute the extraction process for each model and each layer of the model.
    It also displays the extraction graphically.

    Args:
        models (List[Dict[str, Any]]): List of dictionaries containing model data (see Config.get_models).
        extractor_class (Type[Extractor]): Class representing the extractor.
        gpu (str): List of GPU identifiers.
        dataset (Type[Dataset]): Class representing the dataset.

    Return None
    """
    change_mean_std = False

    for model in models:
        logger.info(f'Extraction model: {model["model_name"]}')

        if 'fusion' in model.keys():
            if type(model['fusion']) == list:
                raise ValueError(f'Fusion field cannot be a list!')
            if model['fusion'] not in ['sum', 'concat', 'mul', 'mean']:
                raise NotImplementedError(
                    f'Fusion {model["fusion"]} is not implemented yet! Please select among: ["sum", "concat", "mul", "mean"]')

        extractor = extractor_class(gpu=gpu)

        # set framework
        extractor.set_framework(model['backend'])
        dataset.set_framework(model['backend'])

        # set model
        extractor.set_model(model)
        dataset.set_model(model['model_name'])

        # set preprocessing flag
        dataset.set_preprocessing_flag(model['preprocessing_flag'])
        if isinstance(dataset, VisualDataset):
            if 'preprocessing' in model.keys():  # preprocessing
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
                    raise ValueError(
                        "Mean and std values can be set only for zscore normalization but by the default normalization is None. Please, specify if you want zscore normalization!")

        if isinstance(dataset, VisualTextualDataset):
            dataset.set_model_name(model['model_name'])

        # assess if the extractor is visual or visual-textual in order to properly set the dataset's image processor. 
        if (isinstance(dataset, VisualDataset) or isinstance(dataset, VisualTextualDataset)) and 'transformers' in model['backend']:
            dataset.set_image_processor(extractor._image_processor)

        # execute extractions
        for model_layer in model['output_layers']:
            logger.info(f'Extraction layer: {model["model_name"]}.{model_layer}')

            # set output layer
            extractor.set_output_layer(model_layer)


            if 'tensorflow' not in model['backend']:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=int(model['batch_size']) if 'batch_size' in model.keys() else 1,
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
                        dataset.create_output_file(
                            batch, 
                            extractor_output, 
                            model_layer,
                            fusion=model['fusion'] if 'fusion' in model.keys() else None
                            )
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
        del extractor
        torch.cuda.empty_cache()


class MultimodalFeatureExtractor:

    def __init__(self, config_file_path='./config/config.yml', argv=None):
        """
        Initialize the MultimodalFeatureExtractor.

        Args:
            config_file_path (str): The path to the configuration file (default is './config/config.yml').
                                    It can be either an absolute path or a path relative to the folder of the config file.
            argv (Optional[List[str]]): Additional arguments (default is None).

        Returns:
            None
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
        # logging.add(lambda msg: print(msg, end=""), level="INFO")  # Stream handler for console output # Deprecated

        logger.info('\n' + banner)
        logger.log("WELCOME",
                   '*** Ducho 2.0: Towards a More Up-to-Date Feature Extraction and Processing Framework for Multimodal Recommendation ***')
        logger.log("WELCOME",
                   '*** Brought to you by: SisInfLab, Politecnico di Bari, Italy (https://sisinflab.poliba.it) ***\n')
        self._config = Config(config_file_path, argv)
        # Get the operating system name
        os_name = platform.system()
        # set gpu to use
        os.environ['CUDA_VISIBLE_DEVICES'] = self._config.get_gpu()
        if os_name == 'Darwin':
            if torch.backends.mps.is_available():
                logger.info(
                    f'PYTORCH: Your torch version ({torch.__version__}) and system are compatible with MPS acceleration!')
            else:
                logger.warning(
                    f'PYTORCH: Your torch version ({torch.__version__}) and/or system may not be compatible with MPS acceleration!')
        else:
            logger.info('Checking if CUDA version is compatible with TensorFlow and PyTorch...')
            if len(tf.config.list_physical_devices("GPU")) > 0:
                logger.info(f'TENSORFLOW: Your tf version ({tf.__version__}) is compatible with your CUDA version!')
            else:
                logger.error(
                    f'TENSORFLOW: Your tf version ({tf.__version__}) is not compatible with your CUDA version!')
            if torch.cuda.is_available():
                logger.info(f'PYTORCH: Your torch version ({torch.__version__}) is compatible with your CUDA version!')
            else:
                logger.error(
                    f'PYTORCH: Your torch version ({torch.__version__}) is not compatible with your CUDA version!')

    def execute_extractions(self):
        """
        It executes all the extraction that have to be done
        """
        extractions_dict = self._config.get_extractions()
        for modality, modality_extractions in extractions_dict.items():
            for source, source_extractions in modality_extractions.items():
                self.do_extraction(modality, source)

    def do_extraction(self, modality, source):
        """
        Executes only the generic extraction
        """
        logger.info(f'Extraction on {source} for {modality} modality')

        # instantiate dataset and extractor
        if modality == 'visual_textual':
            # get paths and models
            working_paths = self._config.paths_for_multiple_extraction(source, modality)
            models = self._config.get_models_list(source, modality)
            dataset = getattr(
                importlib.import_module(f"ducho.multimodal.multiple.{modality}.{camel_case(modality)}Dataset"),
                f"{camel_case(modality)}Dataset"
                )(working_paths['input_path'],
                  working_paths['output_path'],
                  columns=self._config.get_columns(modality))
            dataset._textual_dataset.set_type_of_extraction(source)
            extractor = getattr(importlib.import_module(
                f"ducho.multimodal.multiple.{modality}.{camel_case(modality)}FeatureExtractor"),
                f"{camel_case(modality)}FeatureExtractor"
            )
        else:
            # get paths and models
            working_paths = self._config.paths_for_extraction(source, modality)
            models = self._config.get_models_list(source, modality)
            if modality == 'textual':
                dataset = getattr(importlib.import_module(f"ducho.multimodal.{modality}.{camel_case(modality)}Dataset"),
                                  f"{camel_case(modality)}Dataset"
                                  )(working_paths['input_path'],
                                    working_paths['output_path'],
                                    columns=self._config.get_columns(modality))
                dataset.set_type_of_extraction(source)
            else:
                dataset = getattr(importlib.import_module(f"ducho.multimodal.{modality}.{camel_case(modality)}Dataset"),
                                  f"{camel_case(modality)}Dataset"
                                  )(working_paths['input_path'],
                                    working_paths['output_path'])
            extractor = getattr(importlib.import_module(
                f"ducho.multimodal.{modality}.{camel_case(modality)}FeatureExtractor"),
                f"{camel_case(modality)}FeatureExtractor"
            )

        # execute extraction
        logger.info('Extraction is starting...')
        _execute_extraction_from_models_list(models=models,
                                             extractor_class=extractor,
                                             gpu=self._config.get_gpu(),
                                             dataset=dataset)
        logger.success(f'Extraction is complete, it\'s coffee break! ☕️')
