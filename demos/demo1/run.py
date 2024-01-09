from ducho.runner.Runner import MultimodalFeatureExtractor
from datasets import load_dataset
import numpy as np
import os

np.random.seed(42)


def main():
    dataset = load_dataset('ashraq/fashion-product-images-small')
    dataset = np.array(dataset['train'])[np.random.choice(len(dataset['train']), 100, replace=False)].tolist()
    if not os.path.exists('./local/data/demo1/images/'):
        os.makedirs('./local/data/demo1/images/')
    if os.path.exists('./local/data/demo1/descriptions.tsv'):
        os.remove('./local/data/demo1/descriptions.tsv')
    with open(f'./local/data/demo1/descriptions.tsv', 'a') as f:
        f.write('PRODUCT_ID\tdescription\n')
        for file in dataset:
            img = file['image']
            description = f'{file["gender"]} {file["masterCategory"]} ' \
                          f'{file["subCategory"]} {file["articleType"]} ' \
                          f'{file["baseColour"]} {file["season"]} ' \
                          f'{file["year"]} {file["usage"]} ' \
                          f'{file["productDisplayName"]}'
            img.save(f'./local/data/demo1/images/{file["id"]}.jpg')
            f.write(f'{file["id"]}\t{description}\n')
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo1/config_test.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
