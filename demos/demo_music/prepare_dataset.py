# data taken from: https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
import os.path
import time

from tqdm import tqdm

import pandas as pd
import gzip
import requests


def download_image(url, image, save_path):
    response = requests.get(url)
    if response.status_code == 429:
        print('Too many requests, waiting...')
        time.sleep(60)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return image


def parse(path):
    g = gzip.open(path,'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def download_dataset(file_name, save_path):
    base_url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
    response = requests.get(f'{base_url}{file_name}', stream=True)
    if response.status_code == 200:
        with open(f'{save_path}/{file_name}', 'wb') as file:
            bar = tqdm(total=int(response.headers.get('content-length')))
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))


folder = './local/data/demo_music'
name = 'Digital_Music'

core = 5

if not os.path.exists(folder):
    os.makedirs(folder)

# download json
for file_name in [f'reviews_{name}_5.json.gz', f'meta_{name}.json.gz']:
    if not os.path.exists(f'{folder}/{file_name}'):
        download_dataset(file_name, folder)

# load reviews
reviews = getDF(f'{folder}/reviews_{name}_{core}.json.gz')[['reviewerID', 'asin', 'overall']]

# check reviewerID, asin, and overall are not nan and remove duplicates
reviews = reviews[reviews['reviewerID'].notna()]
reviews = reviews[reviews['asin'].notna()]
reviews = reviews[reviews['overall'].notna()]
reviews = reviews.drop_duplicates()

# load meta
meta = getDF(f'{folder}/meta_{name}.json.gz')[['asin', 'description', 'imUrl']]

# check asin, description, and imUrl are not nan
meta = meta[meta['asin'].notna()]
meta = meta[meta['description'].notna()]
meta = meta[meta['description'].str.len() > 0]
meta = meta[meta['imUrl'].notna()]
meta = meta[meta['imUrl'].str.len() > 0]
meta = meta.drop_duplicates()

# merge meta and reviews for 5-core
meta_reviews = pd.merge(reviews, meta, how='inner', on='asin')
meta_reviews.drop_duplicates(inplace=True)
meta = meta_reviews[['asin', 'description', 'imUrl']].drop_duplicates()
reviews = meta_reviews[['reviewerID', 'asin', 'overall']]

# get unavailable items
all_items = set(meta['asin'].tolist())
items_nan_description = set(meta[meta['description'].isna()]['asin'].tolist()).union(set(
    meta[meta['description'].notna()][
        meta[meta['description'].notna()]['description'].str.contains('nan|Nan|NaN|naN|n\/a|N\/A', regex=True)][
        'asin'].tolist()))
items_nan_url = set(meta[meta['imUrl'].isna()]['asin'].tolist()).union(set(meta[meta['imUrl'].notna()][
                                                                            meta[meta['imUrl'].notna()][
                                                                                'imUrl'].str.contains(
                                                                                'nan|Nan|NaN|naN|n\/a|N\/A',
                                                                                regex=True)]['asin'].tolist()))
items_empty_description = set(
    meta[meta['description'].notna()][meta[meta['description'].notna()]['description'].str.len() == 0][
        'asin'].tolist())
items_empty_url = set(
    meta[meta['imUrl'].notna()][meta[meta['imUrl'].notna()]['imUrl'].str.len() == 0]['asin'].tolist())
remaining_items = all_items.difference(items_nan_description).difference(items_nan_url).difference(
    items_empty_description).difference(items_empty_url)
print(f'All items: {len(all_items)}')
print(f'All users: {reviews["reviewerID"].nunique()}')
print(f'All interactions: {len(reviews)}')
print(f'Nan description: {len(items_nan_description)}')
print(f'Nan url: {len(items_nan_url)}')
print(f'Empty description: {len(items_empty_description)}')
print(f'Empty url: {len(items_empty_url)}')

missing_visual = items_nan_url.union(items_empty_url)
missing_textual = items_nan_description.union(items_empty_description)

# save original df
meta.to_csv(f'{folder}/original_meta.tsv', sep='\t', index=None)
reviews.to_csv(f'{folder}/original_reviews.tsv', sep='\t', index=None, header=None)

meta = meta[meta['asin'].isin(all_items.difference(missing_textual))]
reviews = reviews[reviews['asin'].isin(remaining_items)]

print(len(meta[meta['description'].isna()]))
print(len(meta[meta['description'].str.len() == 0]))
print(len(meta[meta['description'].str.contains('nan')]))
print(len(meta[meta['description'].str.contains('Nan')]))
print(len(meta[meta['description'].str.contains('NaN')]))
print(len(meta[meta['description'].str.contains('n\/a')]))
print(len(meta[meta['description'].str.contains('N\/A')]))

print(f'Remaining users: {reviews["reviewerID"].nunique()}')
print(f'Remaining interactions: {len(reviews)}')

# save final df
meta.to_csv(f'{folder}/meta.tsv', sep='\t', index=None)
reviews.to_csv(f'{folder}/reviews.tsv', sep='\t', index=None, header=None)

meta = pd.read_csv(f'{folder}/meta.tsv', sep='\t')
print(len(meta[meta['description'].isna()]))
print(len(meta[meta['description'].str.len() == 0]))


if not os.path.exists(f'{folder}/images/'):
    os.makedirs(f'{folder}/images/')

images = []
with tqdm(total=len(meta)) as t:
    for index, row in meta.iterrows():
        if pd.notna(row['imUrl']):
            output = download_image(row['imUrl'], row['asin'], f'{folder}/images/{row["asin"]}.jpg')
            if output is None:
                missing_visual.add(row['asin'])
            images.append(output)
            t.update()

broken_urls = set([im for im in images if im is None])
print(f'Broken url: {len(broken_urls)}')
remaining_items = remaining_items.intersection(set([im for im in images if im]))
print(f'Remaining items: {len(remaining_items)}')

# save missing items
pd.DataFrame(list(missing_visual)).to_csv(f'{folder}/missing_visual.tsv', sep='\t',
                                        index=None, header=None)
pd.DataFrame(list(missing_textual)).to_csv(f'{folder}/missing_textual.tsv', sep='\t',
                                        index=None,header=None)



