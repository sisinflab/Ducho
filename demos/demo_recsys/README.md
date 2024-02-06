# Demo RecSys

This demo performs visual, textual and multiple feature extraction from items. More precisely, it demonstrates the process of extracting visual and textual features, incorporating custom models as well. Additionally, it showcases the extraction of visual-textual features via multiple modality.

## Run the demo in local
### Create dataset

The following commands are used to create the multimodal dataset for the benchmarking analysis.

Starting from the Amazon Baby catalogue (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) we download product images and textual descriptions, along with recorded user-item interactions. As we retain those items having valid image URLs and non-empty descriptions, the final statistics result in 6,387 users, 19,440 items, and 138,821 interactions (as of the submission time for this paper).

### Download metadata and reviews from the Amazon product data repository

First, we download the reviews (5-core version) and items' metadata: 

```sh
chmod 777 ./demos/demo_recsys/download_amazon.sh
./demos/demo_recsys/download_amazon.sh
```

### Download images/descriptions and create recommendation data

Second, we download the product images and retain only those items having valid URLs and non-empty descriptions:

```sh
python3 ./demos/demo_recsys/prepare_dataset.py
```

Note that the downloaded product images are stored to ```./local/data/demo_recsys/images/```:

```
├── images/
│   ├── 097293751X.jpg
│   ├── 9729375011.jpg
│   ├── B00A0ACI4Q.jpg
│   ├── ...
├── meta.tsv
├── reviews.tsv
```

## Extract multimodal features

Now, we are all setup to extract multimodal features from product images and descriptions.

Let's take a look at the main sections of the configuration file.

### Visual modality

We extract visual features through a pre-trained ResNet50 with `avgpool` as extraction layer and performing **zscore as normalization step**.

Moreover, we extract visual features also through a ResNet50 fine-tuned on the fashion domain. To do so, we **load the custom PyTorch model** we previously created and stored. The other settings are the same as above.
```yaml
visual:
  items:
    input_path: images
    output_path: visual_embeddings
    model: [
      { model_name: ResNet50,  output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch },
      { model_name: ./demos/demo5/MMFashion.pt, output_layers: avgpool, reshape: [224, 224], preprocessing: zscore, backend: torch }
    ]
```

### Textual modality

We extract textual features through a pre-trained SentenceBert hosted on HuggingFace. Note that now we can **explicitly specify the names** of the item column (i.e., storing the items' IDs) and text column (i.e., storing the items' descriptions) as reported in the tsv file.



```yaml
textual:
  items:
    input_path: meta.tsv
    item_column: asin
    text_column: description
    output_path: textual_embeddings
    model: [
      { model_name: sentence-transformers/all-mpnet-base-v2,  output_layers: 1, clear_text: False, backend: sentence_transformers }
    ]
```

### Visual & Textual modalities

We extract the visual and textual features through a pre-trained CLIP multimodal model hosted on HuggingFace. This represents one of the main **novelties introduced with Ducho v2.0.**


```yaml
visual_textual:
  items:
    input_path: {visual: images, textual: meta.tsv}
    item_column: asin
    text_column: description
    output_path: {visual: visual_embeddings, textual: textual_embeddings}
    model: [
      { model_name: openai/clip-vit-base-patch16, backend: transformers, output_layers: 1 }
    ]
```

### Running the multimodal extraction

Finally, we can run the demo for the multimodal feature extraction as indicated in the configuration above:
```sh
PYTHONPATH=. python3 demos/demo_recsys/run.py
```

If everything run smoothly, you should find all extracted multimodal features at the following paths:


```
├── visual_embeddings/
│   ├── torch
│      ├── ResNet50
│         ├── avgpool
│            ├── 097293751X.npy
│            ├── 9729375011.npy
│            ├── B00A0ACI4Q.npy
│            ├── ...
│      ├── MMFashion
│         ├── avgpool
│            ├── 097293751X.npy
│            ├── 9729375011.npy
│            ├── B00A0ACI4Q.npy
│            ├── ...
│   ├── transformers
│      ├── openai
│         ├── clip-vit-base-patch16
│            ├── 1
│               ├── 097293751X.npy
│               ├── 9729375011.npy
│               ├── B00A0ACI4Q.npy
│               ├── ...
├── textual_embeddings/
│   ├── sentence_transformers
│      ├── sentence-transformers
│         ├── all-mpnet-base-v2
│            ├── 1
│               ├── 097293751X.npy
│               ├── 9729375011.npy
│               ├── B00A0ACI4Q.npy
│               ├── ...
│   ├── transformers
│      ├── openai
│         ├── clip-vit-base-patch16
│            ├── 1
│               ├── 097293751X.npy
│               ├── 9729375011.npy
│               ├── B00A0ACI4Q.npy
│               ├── ...
```


## Run the demo with Docker

This is the command to run:

```sh
docker compose run prepare_dataset
```

```sh
docker compose run extract_features
```

## Outputs

Note that Ducho saves all extracted files, along with the log files, into the folder ./local/data.

For this demo, you may find them at:

- visual embeddings: ./local/data/demo_recsys/visual_embeddings
- textual embeddings: ./local/data/demo_recsys/textual_embeddings
- logs: ./local/logs

## Run multimodal recommendation benchmarks with the Elliot framework

Now we can work with the extracted multimodal features to run any multimodal recommendation benchmarks. For the sake of this demo, we use our recommendation framework **Elliot**, where six multimodal recommendation systems are currently re-implemented and ready to be trained/tested on custom datasets and hyper-parameter settings.

**Useful references for Elliot**

* Paper: https://dl.acm.org/doi/10.1145/3404835.3463245
* GitHub: https://github.com/sisinflab/elliot
* Documentation: https://elliot.readthedocs.io/en/latest/index.html

**Elliot for Multimodal Recommendation**
* Paper: https://arxiv.org/abs/2309.05273
* GitHub: https://github.com/sisinflab/Formal-MultiMod-Rec

For the sake of this demo, we benchmark the recommendation performance of **VBPR**, **BM3**, and **FREEDOM**, in three multimodal extraction settings.

* \[STANDARD\] Visual: ResNet50, Textual: SentenceBert
* **\[NEW\]** Visual & Textual: CLIP
* **\[NEW\]** Visual: MMFashion, Textual: SentenceBert

### Install Elliot

First, we clone the repository from GitHub and install all needed dependencies:

```sh
cd ../
git clone https://github.com/sisinflab/Formal-Multimod-Rec.git
cd Formal-Multimod-Rec/
env CUBLAS_WORKSPACE_CONFIG=:16:8
pip install -r requirements_torch_geometric_colab.txt
```
### Split the recommendation data

Second, we split the recommendation data we created in the previous steps:

```sh
cp ../Ducho/local/data/demo_recsys/reviews.tsv data/baby/
cp -r ../Ducho/local/data/demo_recsys/visual_embeddings/ data/baby/
cp -r ../Ducho/local/data/demo_recsys/textual_embeddings/ data/baby/
python3 run_split.py
```

### Train and test multimodal recommender systems

Now, we are all set to train and test the three selected multimodal recommendation systems:

```sh
python3 run_benchmarking.py --setting 1
```

Change the `setting` parameter according to the multimodal extractor setting you want to run. Note that this might take some time, as we are exploring 10 hyper-parameter configurations for each of the multimodal recommendation systems involved!

For you convenience, we report the recommendation results here (just as they appear in the resource paper):

| Extractors | Models  | Recall | Precision | nDCG  | HR    |
|------------|---------|--------|-----------|-------|-------|
| Visual: ResNet50 <br> Textual: SentenceBert | VBPR    | 0.0613 | 0.0055    | 0.0309 | 0.1031 |
|                                             | BM3     | 0.0850 | 0.0076    | 0.0426 | 0.1420 |
|                                             | FREEDOM | 0.0935 | 0.0083    | 0.0465 | 0.1537 |
| Visual & Textual: CLIP                       | VBPR    | 0.0630 | 0.0057    | 0.0313 | 0.1068 |
|                                             | BM3     | 0.0851 | 0.0076    | 0.0428 | 0.1425 |
|                                             | FREEDOM | 0.0704 | 0.0064    | 0.0351 | 0.1187 |
| Visual: MMFashion <br> Textual: SentenceBert | VBPR    | 0.0619 | 0.0055    | 0.0309 | 0.1027 |
|                                             | BM3     | 0.0847 | 0.0076    | 0.0420 | 0.1423 |
|                                             | FREEDOM | 0.0932 | 0.0083    | 0.0465 | 0.1542 |



### Google Colab

Follow this link: https://colab.research.google.com/drive/1vPUALePlrjv4rfSn6CX2zMkpH2Xrw_cp



