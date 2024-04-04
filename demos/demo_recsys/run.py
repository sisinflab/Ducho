from ducho.runner.Runner import MultimodalFeatureExtractor
import torch
import torchvision.models as models
from collections import OrderedDict
import os
import gdown
import numpy as np
import random

def set_seed(seed = 42):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


def main():
    set_seed()
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo_recsys/config.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    if not os.path.exists('./demos/demo_recsys/MMFashion.pt'):
        gdown.download(f'https://drive.google.com/uc?id=1LmC4aKiOY3qmm9qo6RNDU5v_o-xDCAdT', './demos/demo_recsys/MMFashion.pth', quiet=False)
        model = torch.load('./demos/demo_recsys/MMFashion.pth', map_location=torch.device('cpu'))
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
        torch.save(resnet50, './demos/demo_recsys/MMFashion.pt')
        os.remove('./demos/demo_recsys/MMFashion.pth')
    main()
