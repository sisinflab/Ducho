import torch
from transformers import pipeline
from PIL import Image
import requests

model_name = 'kakaobrain/align-base'
# model_name = 'openai/clip-vit-base-patch32'
# model_name = 'google/owlvit-base-patch32'
# model_name = 'uclanlp/visualbert-vqa-coco-pre'

pipe = pipeline(task='feature-extraction',
                model=model_name,
                image_processor=model_name,
                tokenizer=model_name
                )

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
candidate_labels = ["an image of a cat", "an image of a dog"]

print(type(image))

model = pipe.model
image_processor = pipe.image_processor
tokenizer = pipe.tokenizer

a = tokenizer(candidate_labels)
b = image_processor(image)

a = {k: torch.tensor(v) for k, v in a.items()}
b = {k: torch.tensor(v) for k, v in b.items()}

a.update(b)

outputs = model(**a)
print(outputs.keys())
print(outputs['text_embeds'].shape)
print(outputs['image_embeds'].shape)

print(outputs['text_model_output'].pooler_output.shape)
print(outputs['vision_model_output'].pooler_output.shape)



