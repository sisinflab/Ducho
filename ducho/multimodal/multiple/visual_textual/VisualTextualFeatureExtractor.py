from transformers import pipeline
import torch
from ducho.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


class VisualTextualFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Textual extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        Args:
             gpu: String on which is explained which gpu to use. '-1' -> cpu
        """
        self._pipeline = None
        self._tokenizer = None
        self._image_processor = None
        self._task = 'feature_extraction'
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model: is the dictionary of the configuration for the model
        Returns: nothing but it initializes the protected model and tokenizer attributes, later used for extraction
        """
        model_name = model['name']
        tokenizer_name = model['tokenizer'] if model['tokenizer'] else model['name']
        image_processor_name = model['image_processor'] if model['image_processor'] else model['name']

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
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        Args:
            sample_input: a tuple containing the image and the text to preprocess
        Returns:
             two numpy arrays that will be put in two .npy file calling the right Dataset Class' method
        """

        image, text = sample_input
        preprocessed_text = self._tokenizer(text[0])
        preprocessed_image = self._image_processor(image, do_rescale=False)

        preprocessed_text = {k: torch.tensor(v).unsqueeze(dim=0).to(self._device) for k, v in preprocessed_text.items()}
        preprocessed_image = {k: torch.tensor(v).to(self._device) for k, v in preprocessed_image.items()}

        preprocessed_text.update(preprocessed_image)

        outputs = self._model(**preprocessed_text)

        return outputs['image_embeds'].detach().cpu().numpy(), outputs['text_embeds'].detach().cpu().numpy()




