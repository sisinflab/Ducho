from ducho.runner.Runner import MultimodalFeatureExtractor
import torch
import torchvision.models as models
from collections import OrderedDict
import os


def main():
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo_recsys/config.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    if not os.path.exists('./demos/demo_recsys/customResNet50.pth'):
        model = torch.load('./demos/demo_recsys/model_best.pth', map_location=torch.device('cpu'))
        resnet50 = models.resnet50(pretrained=False)
        state_dict = model['state_dict']
        od = OrderedDict()
        for key, value in state_dict.items():
            if 'backbone' in key:
                od[key.replace('backbone.', '')] = value
            else:
                break
        fc = torch.nn.Linear(in_features=2048, out_features=1000)
        od['fc.weight'] = fc.weight
        od['fc.bias'] = fc.bias
        resnet50.load_state_dict(od)
        torch.save(resnet50, './demos/demo_recsys/customResNet50.pt')
    main()
