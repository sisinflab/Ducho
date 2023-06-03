import os
import logging
import datetime

if not os.path.exists('./local/logs/'):
    os.makedirs('./local/logs/')

log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%Y-%m-%d-%H:%M:%S',
    handlers=[
        logging.FileHandler(filename=f'./local/logs/{log_file}.log'),
        logging.StreamHandler()
    ]
)


from src.runner.Runner import MultimodalFeatureExtractor
from torchvision.datasets import FashionMNIST
import shutil
import numpy as np
import torch


def main():
    dataset = FashionMNIST(root='./local/data/demo1/', download=True)
    dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 100, replace=False))
    if not os.path.exists('./local/data/demo1/images/'):
        os.makedirs('./local/data/demo1/images/')
    for idx, (img, _) in enumerate(dataset):
        img.save('./local/data/demo1/images/{:03d}.jpg'.format(idx))
    shutil.rmtree('./local/data/demo1/FashionMNIST/')
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo1/demo1.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
