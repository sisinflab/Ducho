from ducho.runner.Runner import MultimodalFeatureExtractor
from datasets import load_dataset
import numpy as np
import os
from torch import nn

np.random.seed(42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def main():
    dataset = load_dataset('ashraq/fashion-product-images-small')
    dataset = np.array(dataset['train'])[np.random.choice(len(dataset['train']), 100, replace=False)].tolist()
    if not os.path.exists('./local/data/demo4/images/'):
        os.makedirs('./local/data/demo4/images/')
    if os.path.exists('./local/data/demo4/descriptions.tsv'):
        os.remove('./local/data/demo4/descriptions.tsv')
    with open(f'./local/data/demo4/descriptions.tsv', 'a') as f:
        f.write('PRODUCT_ID\tdescription\n')
        for file in dataset:
            img = file['image']
            description = f'{file["gender"]} {file["masterCategory"]} ' \
                          f'{file["subCategory"]} {file["articleType"]} ' \
                          f'{file["baseColour"]} {file["season"]} ' \
                          f'{file["year"]} {file["usage"]} ' \
                          f'{file["productDisplayName"]}'
            img.save(f'./local/data/demo4/images/{file["id"]}.jpg')
            f.write(f'{file["id"]}\t{description}\n')
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo4/config_test.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
