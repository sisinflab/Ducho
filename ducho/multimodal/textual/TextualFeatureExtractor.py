from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
from ducho.internal.father_classes.FeatureExtractorFather import FeatureExtractorFather


class TextualFeatureExtractor(FeatureExtractorFather):
    """
        This class represents the Textual Feature Extractor utilized for feature extraction.
    """
    def __init__(self, gpu='-1'):
        """
        This function carries out Textual Feature Extraction, requiring the 'model_name', 'framework', and 'output_layer'.

        Args:
            gpu: A string indicating the GPU to be used. '-1' specifies the CPU.

        Returns:
            None
        """
        self._pipeline = None
        self._tokenizer = None
        self.output_layer = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        This procedure facilitates the configuration of the Textual Feature Extractor model using YAML specifications.

        Args:
            model: The row of the YAML file containing the user's specifications.

        Returns:
            None
        """
        model_name = model['model_name']

        if 'transformers' in self._backend_libraries_list:
            tokenizer_name = model['tokenizer_name'] if 'tokenizer_name' in model.keys() else model['model_name']
            built_pipeline = pipeline(task='feature-extraction', model=model_name, tokenizer=tokenizer_name, framework='pt', device=self._device)
            self._model = built_pipeline.model
            self._tokenizer = built_pipeline.tokenizer
            self.output_layer = model['output_layers'][0] if 'output_layers' in model.keys() else 'pooler_output'
        elif 'sentence_transformers' in self._backend_libraries_list:
            self._model = SentenceTransformer(model_name)

    def extract_feature(self, sample_input):
        """
        This function extracts features from the input text. Prior to calling this function, the framework,
        model, and layer have to be configured using their respective set methods.

        Args:
            sample_input: The preprocessed textual data.

        Returns:
            A numpy array representing the extracted features, which will be stored in a .npy file using the appropriate method of the Dataset Class.
        """
        if 'transformers' in self._backend_libraries_list:
            model_input = self._tokenizer.batch_encode_plus(sample_input[0], return_tensors="pt", padding=True, truncation=True)
            model_input = {k: torch.tensor(v).to(self._device) for k, v in model_input.items()}
            model_output = getattr(self._model(**model_input), self.output_layer)
            return model_output.detach().cpu().numpy()
        elif 'sentence_transformers' in self._backend_libraries_list:
            return self._model.encode(sentences=sample_input[0])