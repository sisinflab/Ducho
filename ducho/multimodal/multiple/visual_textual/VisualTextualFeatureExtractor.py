from transformers import pipeline
import torch
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather
from transformers import AutoTokenizer

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
            self._tokenizer = built_pipeline.tokenizer if not type(built_pipeline.tokenizer) is str else AutoTokenizer.from_pretrained(tokenizer_name)
            self._image_processor = built_pipeline.image_processor

            # if type(model['output_layers']) is not str and type(model['output_layers']) is not list:
            #     self._vision_output = ['image_embeds']
            #     self._text_output = ['text_embeds']
            # else:
            #     self._vision_output = model['output_layers'][0].split('.')
            #     self._text_output = model['output_layers'][1].split('.')

            if all(isinstance(x, int) for x in model['output_layers']):
                self._vision_output = ['image_embeds']
                self._text_output = ['text_embeds']
            elif all(isinstance(x, str) for x in model['output_layers']):
                if len(model['output_layers']) > 1:
                    self._vision_output = model['output_layers'][0].split('.')
                    self._text_output = model['output_layers'][1].split('.')
                else:
                    self._vision_output = model['output_layers'][0].split('.')
                    self._text_output = model['output_layers'][0].split('.')
            else:
                raise ValueError('Layers must be of the same type!')

        else:
            raise NotImplemented('This feature extractor has not been added yet!')

    @torch.no_grad
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
        
        try:
            preprocessed_text = self._tokenizer.batch_encode_plus(text[0], return_tensors="pt", padding='max_length', truncation=True)
        except ValueError:
            preprocessed_text = self._tokenizer.batch_encode_plus(text[0], return_tensors="pt", padding=True, truncation=True)

        # converting the input image tensor - outcome of the pre-processor - in a set.
        preprocessed_image = {'pixel_values': image[0]}

        # preprocessed_text = {k: torch.tensor(v).unsqueeze(dim=0).to(self._device) for k, v in preprocessed_text.items()}
        preprocessed_text = {k: torch.tensor(v).to(self._device) for k, v in preprocessed_text.items()}
        preprocessed_image = {k: torch.tensor(v).to(self._device) for k, v in preprocessed_image.items()}

        preprocessed_text.update(preprocessed_image)

        outputs = self._model(**preprocessed_text)
        
        if len(self._vision_output) > 1:
            temp_output_vis = outputs
            for name in self._vision_output:
                temp_output_vis = getattr(temp_output_vis, name)
        else:
            temp_output_vis = getattr(outputs, self._vision_output[0])


        if len(self._text_output) > 1:
            temp_output_text = outputs
            for name in self._text_output:
                temp_output_text = getattr(temp_output_text, name)
        else:
            temp_output_text = getattr(outputs, self._text_output[0])

        vis_output = temp_output_vis.detach().cpu().numpy()
        text_output = temp_output_text.detach().cpu().numpy()

        return vis_output, text_output




