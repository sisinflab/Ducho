## Demo 2: audio and textual feature extraction from items

This demo shows how to extract audio and textual features from songs and their genre labels, respectively.

Dataset source: https://huggingface.co/datasets/lewtun/music_genres_small

Please note that the original dataset has been sampled for the sake of this demo (i.e., 10 songs are randomly sampled - the low number is due to the heavy computational cost of the pretrained models involved).

## Run the demo

### Local

Assuming the virtual environment has been correctly created and activated, this is the command to run:

```
PYTHONPATH=. python3 demos/demo2/run.py
```

### Docker

This is the command to run:

```
docker compose run demo2
```

### Google Colab

Follow this link: https://colab.research.google.com/drive/1ouKkdxOObOL0BI00r0c157oNRqwxqTgt#scrollTo=5uUXfKpgdkAA&line=7&uniqifier=1

## Outputs

For this demo, you may find output files at:

- audio output embeddings: ./local/data/demo2/audio_embeddings
- textual output embeddings: ./local/data/demo2/textual_embeddings
- logs: ./local/logs
