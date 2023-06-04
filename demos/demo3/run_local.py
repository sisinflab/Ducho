import random
from src.runner.Runner import MultimodalFeatureExtractor
import numpy as np
import os
import shutil

np.random.seed(42)
random.seed(42)


def main():
    if not os.path.exists('./local/data/demo3/songs/'):
        os.makedirs('./local/data/demo3/songs/')
    print('Dataset: MUSDB18-HQ - a corpus for music separation')
    print('Visit the original website at: https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems')
    print('\nPlease, download the music dataset at: https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1')
    print('It might take some time...')
    print('Once you are done, move the file to ./local/data/demo3/songs/')
    if not os.path.exists('./local/data/demo3/songs/musdb18hq.zip'):
        if len(os.listdir('./local/data/demo3/songs/')) != 100:
            print('The file is not to the right path...')
            exit(1)
    else:
        print('Unzipping the file. This could take some time...')
        shutil.unpack_archive('./local/data/demo3/songs/musdb18hq.zip', './local/data/demo3/songs/')
        os.remove('./local/data/demo3/songs/musdb18hq.zip')
        print('Unzipping complete!')
        songs = os.listdir('./local/data/demo3/songs/train/')
        random.shuffle(songs)
        for idx, s in enumerate(songs[:100]):
            shutil.move(f'./local/data/demo3/songs/train/{s}', f'./local/data/demo3/songs/{idx}.wav')
        shutil.rmtree('./local/data/demo3/songs/train/')
        shutil.rmtree('./local/data/demo3/songs/test/')
        os.remove('./local/data/demo3/songs/README.md')
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo3/demo3.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
