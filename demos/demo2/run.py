from ducho.runner.Runner import MultimodalFeatureExtractor
from datasets import load_dataset
import numpy as np
import os

np.random.seed(1234)


def main():
    dataset = load_dataset('lewtun/music_genres_small')
    dataset = np.array(dataset['train'].to_list())[np.random.choice(len(dataset['train']), 10, replace=False)].tolist()
    if not os.path.exists('./local/data/demo2/songs/'):
        os.makedirs('./local/data/demo2/songs/')
    with open(f'./local/data/demo2/genres.tsv', 'a') as f:
        f.write('SONG_ID\tdescription\n')
        for file in dataset:
            f.write(f'{file["song_id"]}\t{file["genre"]}\n')
            with open(f'./local/data/demo2/songs/{file["song_id"]}.wav', 'wb') as a:
                a.write(file['audio']['bytes'])
    extractor_obj = MultimodalFeatureExtractor(config_file_path='./demos/demo2/config.yml')
    extractor_obj.execute_extractions()


if __name__ == '__main__':
    main()
