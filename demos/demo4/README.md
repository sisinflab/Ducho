# Demo 1: visual and textual feature extraction from items

This demo shows how to extract visual and textual features from fashion product images and descriptions, respectively.

Dataset source: https://huggingface.co/datasets/ashraq/fashion-product-images-small

Please note that the original dataset has been sampled for the sake of this demo (i.e., 100 fashion items are randomly sampled).

## Run the demo

### Local

Assuming the virtual environment has been correctly created and activated, this is the command to run:

```sh
PYTHONPATH=. python3 demos/demo1/run.py
```

### Docker

This is the command to run:

```sh
docker compose run demo1
```

### Google Colab

Follow this link: https://colab.research.google.com/drive/1ouKkdxOObOL0BI00r0c157oNRqwxqTgt#scrollTo=TAGeT3ONZAGU&line=7&uniqifier=1

## Outputs

Note that Ducho saves all extracted files, along with the log files, into the folder ./local/data.

For this demo, you may find them at:

- visual embeddings: ./local/data/demo1/visual_embeddings
- textual embeddings: ./local/data/demo1/textual_embeddings
- logs: ./local/logs


