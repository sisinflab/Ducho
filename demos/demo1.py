from src.runner.Runner import MultimodalFeatureExtractor
from torchvision.datasets import FashionMNIST
import shutil
import numpy as np
import torch
import os


def main(argv):
    dataset = FashionMNIST(root='./data/demo1/', download=True)
    dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), 100, replace=False))
    if not os.path.exists('./data/demo1/images/'):
        os.makedirs('./data/demo1/images/')
    for idx, (img, _) in enumerate(dataset):
        img.save('./data/demo1/images/{:03d}.jpg'.format(idx))
    shutil.rmtree('./data/demo1/FashionMNIST/')
    extractor_obj = MultimodalFeatureExtractor(argv=argv)
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main(['--dataset_path=./data/demo1', '--gpu_list=0'])
