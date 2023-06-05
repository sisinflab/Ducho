from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import FeatureExtractionPipeline
from transformers import PreTrainedModel
from operator import attrgetter
# import transformers.pipelines.

import torch
# import torchtext
from ducho.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


# def flatten_model(previous_name, model, layer_list):
#     current_list = list(model.named_children())
#     if current_list:
#         for current_name, value in current_list:
#             next_name = f'{previous_name}.{current_name}'
#             flat = flatten_model(next_name, value, layer_list)
#             if not isinstance(flat, list):
#                 layer_list.append(flat)
#     else:
#         return previous_name.replace('x.', '')
#     return layer_list


class TextualCnnFeatureExtractor(CnnFeatureExtractorFather):
    def __init__(self, gpu='-1'):
        """
        It does Textual extraction. It is needed also to give the model name, the framework and the output_layer. You can
        later change one of them as needed.
        :param gpu: gpu: String on which is explained which gpu to use. '-1' -> cpu
        """
        self._pipeline = None
        self._tokenizer = None
        super().__init__(gpu)

    def set_model(self, model):
        """
        Args:
            model_name: is the name of the model to use as a String.
                        NOTE: in this case we are using transformers so the model name have to be in its list.
                        Since we are using transformers here, it is needed also to point the repo so: 'repo/model'
        Returns: nothing but it initializes the protected model and tokenizer attributes, later used for extraction
        :param model:
        """
        model_name = model['name']
        if 'task' in model.keys():
            model_task = model['task']
        if 'transformers' in self._framework_list:
            built_pipeline = pipeline(model_task, model=model_name)
            self._model = built_pipeline.model
            self._tokenizer = built_pipeline.tokenizer
        elif 'sentence_transformers' in self._framework_list:
            self._model = SentenceTransformer(model_name)

    def extract_feature(self, sample_input):
        """
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        :param sample_input: the String in input to process
        :return: a numpy array that will be put in a .npy file calling the right Dataset Class' method
        """
        if 'transformers' in self._framework_list:
            model_input = self._tokenizer.encode_plus(sample_input, truncation=True, return_tensors="pt")
            model_output = list(self._model.children())[-self._output_layer](**model_input, output_hidden_states=True).pooler_output
            return model_output.detach().numpy()
        elif 'sentence_transformers' in self._framework_list:
            return self._model.encode(sentences=sample_input)




