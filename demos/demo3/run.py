from src.runner.Runner import MultimodalFeatureExtractor
from datasets import load_dataset
import numpy as np
import pandas as pd
import os

np.random.seed(42)


def main():
    product_dict = {}
    dataset = load_dataset('amazon_us_reviews', 'Personal_Care_Appliances_v1_00')
    dataset = np.array(dataset['train'].to_list())[np.random.choice(len(dataset['train']), 100, replace=False)].tolist()
    if not os.path.exists('./local/data/demo3/'):
        os.makedirs('./local/data/demo3/')
    with open(f'./local/data/demo3/reviews.tsv', 'a') as f1:
        f1.write('USER_ID\tPRODUCT_ID\tREVIEW\n')
        for idx, d in enumerate(dataset):
            if not d["product_id"] in product_dict.keys():
                product_dict[d["product_id"]] = d["product_title"] + " " + d["product_category"]
            f1.write(f'{d["customer_id"]}\t{d["product_id"]}\t{d["review_body"]}\n')
    pd.DataFrame(product_dict.items(), columns=['PRODUCT_ID', 'DESCRIPTION']).to_csv(
        f'./local/data/demo3/descriptions.tsv', sep='\t', index=None)
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo3/config.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
