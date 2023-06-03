from src.runner.Runner import MultimodalFeatureExtractor
print('ciao')
from torchvision.datasets import FashionMNIST
import shutil
import numpy as np
import torch
import os


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
