## Demo 1: visual and textual feature extraction from items

This demo shows how to extract visual and textual features from fashion product images and descriptions, respectively.

Dataset source: https://huggingface.co/datasets/ashraq/fashion-product-images-small

Please note that the original dataset has been sampled for the sake of this demo (i.e., 100 fashion items are randomly sampled).

In the following, we show how to run this demo.

### Local

Assuming the virtual environment has been correctly created and activated, this is the command to run:

```
PYTHONPATH=. python3 demos/demo1/run.py
```

### Docker

This is the command to run:

```
docker compose run demo1
```
