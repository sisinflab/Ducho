from transformers import pipeline
from transformers import FeatureExtractionPipeline
from transformers import PreTrainedModel
# import transformers.pipelines.

import torch
# import torchtext
from src.internal.father_classes.CnnFeatureExtractorFather import CnnFeatureExtractorFather


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
        model_task = model['task']
        if 'transformers' in self._framework_list:
            built_pipeline = pipeline(model_task, model=model_name)
            self._model = built_pipeline.model
            self._tokenizer = built_pipeline.tokenizer
            # self._pipeline = built_pipeline
            # sentiment_pipeline = pipeline(model=model_name)
            # model = list(sentiment_pipeline.model.children())[-3]
            # model.eval()
            # model.to(self._device)
            # self._model = model
            # self._tokenizer = sentiment_pipeline.tokenizer

            # extraction_pipeline = pipeline("sentiment-analysis", model="bert-base-uncased")
            # self._model = extraction_pipeline

    def extract_feature(self, sample_input):
        """
        It does extract the feature from the input. Framework, model and layer HAVE TO be already set with their set
        methods.
        :param sample_input: the String in input to process
        :return: a numpy array that will be put in a .npy file calling the right Dataset Class' method
        """
        if 'transformers' in self._framework_list:
            model_input = self._tokenizer(sample_input, return_tensors="pt")
            model_output = self._model(**model_input, output_hidden_states=True)
            layer_output = model_output.hidden_states[self._output_layer]
            return layer_output.detach().numpy()

            # output = self._tokenizer.encode_plus(sample_input, return_tensors="pt").to(self._device)
            # return self._model(**output.to(self._device)).pooler_output.detach().cpu().numpy()

            # output = self._model(sample_input)
            # layer = output[0]["hidden_states"][self._output_layer]
            # return layer.detach().numpy()
            # extraction_pipeline = pipeline("sentiment-analysis", model="bert-base-uncased")
            # output = extraction_pipeline(sample_input)
            # print(output)

            # model = PreTrainedModel("bert-base-uncased")
            # the_pipeline = FeatureExtractionPipeline(model='')
            # model = pipeline("feature-extraction", model="bert-base-uncased")
            # print('heo')

            # classifier = pipeline("question-answering", model="stevhliu/my_awesome_model")
            # model = classifier.model
            # tokenizer = classifier.tokenizer
            # inputt = tokenizer(sample_input, return_tensors="pt")
            # output = model(**inputt, output_hidden_states=True)




