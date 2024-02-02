# data taken from: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
import os.path
from tqdm import tqdm

import pandas as pd
import gzip
import requests


def download_image(url, image, save_path):
  response = requests.get(url)
  if response.status_code == 200:
    with open(save_path, 'wb') as file:
      file.write(response.content)
    return image


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)


def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


folder = './local/data/demo_recsys'
name = 'Baby'
core = 5

# load reviews
reviews = getDF(f'{folder}/reviews_{name}_{core}.json.gz')[['reviewerID', 'asin', 'overall']]

# check reviewerID, asin, and overall are not nan
reviews = reviews[reviews['reviewerID'].notna()]
reviews = reviews[reviews['asin'].notna()]
reviews = reviews[reviews['overall'].notna()]

# load meta
meta = getDF(f'{folder}/meta_{name}.json.gz')[['asin', 'description', 'imUrl']]

# check asin, description, and imUrl are not nan
meta = meta[meta['asin'].notna()]
meta = meta[meta['description'].notna()]
meta = meta[meta['description'].str.len() > 0]
meta = meta[meta['imUrl'].notna()]
meta = meta[meta['imUrl'].str.len() > 0]

meta_reviews = pd.merge(reviews, meta, how='inner', on='asin')
meta_reviews.drop_duplicates(inplace=True)

meta = meta_reviews[['asin', 'description', 'imUrl']].drop_duplicates()
reviews = meta_reviews[['reviewerID', 'asin', 'overall']]

if not os.path.exists(f'{folder}/images/'):
    os.makedirs(f'{folder}/images/')

images_ok = []
with tqdm(total=len(meta)) as t:
    for index, row in meta.iterrows():
        images_ok.append(download_image(row['imUrl'], row['asin'], f'{folder}/images/{row["asin"]}.jpg'))
        t.update()

# remove meta and reviews for which there is no available image
meta = meta[meta['asin'].isin(images_ok)]
reviews = reviews[reviews['asin'].isin(images_ok)]

# save final df
meta.to_csv(f'{folder}/meta.tsv', sep='\t', index=None)
reviews.to_csv(f'{folder}/reviews.tsv', sep='\t', index=None, header=None)



