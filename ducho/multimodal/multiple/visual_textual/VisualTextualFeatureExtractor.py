from transformers import pipeline
import torch
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather


class VisualTextualFeatureExtractor(FeatureExtractorFather):
    """
    This class represents the Visual-Textual Feature Extractor utilized for feature extraction.
    """
    def __init__(self, gpu='-1'):
        """
        This function carries out Visual-Textual Feature Extraction, requiring the 'model_name', 'framework', and 'output_layer'.

        Args:
             gpu: A string indicating the GPU to be used. '-1' specifies the CPU.

        Returns:
            None
        """
        self._pipeline = None
        self._tokenizer = None
        self._image_processor = None
        self._task = 'feature_extraction'
        super().__init__(gpu)

    def set_model(self, model):
        """
        This procedure facilitates the configuration of the Visual-Textual Feature Extractor model using YAML specifications.

        Args:
            model: The row of the YAML file containing the user's specifications.

        Returns:
            None
        """
        model_name = model['model_name']
        tokenizer_name = model['tokenizer_name'] if 'tokenizer_name' in model.keys() else model['model_name']
        image_processor_name = model['image_processor_name'] if 'image_processor_name' in model.keys() else model['model_name']

        if 'transformers' in self._backend_libraries_list:
            built_pipeline = pipeline(
                task='feature-extraction',
                model=model_name,
                image_processor=image_processor_name,
                tokenizer=tokenizer_name,
                device=self._device
                )

            self._model = built_pipeline.model
            self._tokenizer = built_pipeline.tokenizer
            self._image_processor = built_pipeline.image_processor
        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    def extract_feature(self, sample_input):
        """
        This function extracts features from the input image and textual data. Prior to calling this function, the framework,
        model, and layer have to be configured using their respective set methods.

        Args:
            sample_input: The preprocessed data.

        Returns:
            Two numpy array representing the extracted features, which will be stored in two .npy files using the appropriate method of the Dataset Class.
        """

        image, text = sample_input
        preprocessed_text = self._tokenizer(text[0], truncation=True)
        preprocessed_image = self._image_processor(image)

        preprocessed_text = {k: torch.tensor(v).unsqueeze(dim=0).to(self._device) for k, v in preprocessed_text.items()}
        preprocessed_image = {k: torch.tensor(v).to(self._device) for k, v in preprocessed_image.items()}

        preprocessed_text.update(preprocessed_image)

        outputs = self._model(**preprocessed_text)

        return (outputs.image_embeds.detach().cpu().numpy(),
                outputs.text_embeds.detach().cpu().numpy())




