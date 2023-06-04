from src.runner.Runner import MultimodalFeatureExtractor
from datasets import load_dataset
import numpy as np
import os

np.random.seed(42)


def main():
    dataset = load_dataset('ashraq/fashion-product-images-small')
    dataset = np.array(dataset['train'])[np.random.choice(len(dataset['train']), 100, replace=False)].tolist()
    if not os.path.exists('./local/data/demo2/images/'):
        os.makedirs('./local/data/demo2/images/')
    if not os.path.exists('./local/data/demo2/descriptions/'):
        os.makedirs('./local/data/demo2/descriptions/')
    with open(f'./local/data/demo2/descriptions/descriptions.tsv', 'a') as f:
        f.write('PRODUCT_ID\tdescription\n')
        for idx, file in enumerate(dataset):
            img = file['image']
            description = f'{file["gender"]} {file["masterCategory"]} ' \
                          f'{file["subCategory"]} {file["articleType"]} ' \
                          f'{file["baseColour"]} {file["season"]} ' \
                          f'{file["year"]} {file["usage"]} ' \
                          f'{file["productDisplayName"]}'
            img.save(f'./local/data/demo2/images/{idx}.jpg')
            f.write(str(idx) + '\t' + description + '\n')
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo2/demo2.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
