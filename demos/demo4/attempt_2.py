import torch
from transformers import pipeline, FeatureExtractionPipeline, DeiTFeatureExtractor
from transformers import AlignProcessor, AlignModel
from PIL import Image
import requests

model_name = 'kakaobrain/align-base'
# model_name = 'openai/clip-vit-base-patch32'
#model_name = 'google/owlvit-base-patch32'
#model_name = 'uclanlp/visualbert-vqa-coco-pre'
# model_name = 'facebook/deit-tiny-patch16-224'
# model_name = 'facebook/deit-base-distilled-patch16-224'

pipe = pipeline(task='feature-extraction',
                model=model_name,
                image_processor=model_name,
                tokenizer=model_name
                # feature_extractor=model_name
                )


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

model = pipe.model
image_processor = pipe.image_processor
tokenizer = pipe.tokenizer

a = tokenizer(candidate_labels)
b = image_processor(image)
# print(b.keys())

a = {k: torch.tensor(v) for k, v in a.items()}
b = {k: torch.tensor(v) for k, v in b.items()}

a.update(b)

outputs = model(**a)
print(outputs.keys())
# print(outputs['text_embeds'].shape)
# print(outputs['image_embeds'].shape)
############################

image_feature_pipe = outputs['image_embeds']
text_feature_pipe = outputs['text_embeds']


# feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
# #feature_extractor = pipe.feature_extractor
# # Create a feature extraction pipeline
# feature_extraction_pipeline = FeatureExtractionPipeline(model=model, feature_extractor=feature_extractor)

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")


def extract_last_layer(image_path):
    # Load and preprocess the image
    inputs = processor(images=image_path, return_tensors="pt", text=candidate_labels)
    # Forward pass through the model
    outputs = model(**inputs)
    # print(outputs.keys())
    # Extract the last layer (last_hidden_state)
    # last_layer = outputs.pooler_output
    image_emb, text_emb = outputs['image_embeds'], outputs['text_embeds']
    last_layer = (image_emb, text_emb)
    return last_layer


#image_path_to_process = Image.open("./test.jpg")

# result = extract_last_layer(image)

# The result is a PyTorch tensor containing the last layer of the model for the given image
# print(result)
# print(result.shape, image_feature_pipe.shape)
# print(torch.all(result[0] == image_feature_pipe))
# print(torch.all(result[1] == text_feature_pipe))
# print(result[1].shape, text_feature_pipe.shape)

from transformers import AutoTokenizer, BertModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(candidate_labels, return_tensors="pt")
outputs_original = model(**inputs)

print(outputs.keys())
model_name = 'bert-base-uncased'

pipe = pipeline(task='feature-extraction',
                model=model_name,
                tokenizer=model_name
                # feature_extractor=model_name
                )

tokenizer = pipe.tokenizer

a = tokenizer(candidate_labels)

a = {k: torch.tensor(v) for k, v in a.items()}
model = pipe.model

outputs = model(**a)

print(outputs.keys())



print(torch.all(outputs['pooler_output'] == outputs_original['pooler_output']))