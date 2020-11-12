""" Add images into a pandas Dataframe
"""
from pathlib import Path

import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import tqdm
from pathlib import Path
import shutil
import os

dirpath = Path('data/dataset')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
os.mkdir('data/dataset')
datasetPath = Path('data/dataset/dataset.zip')
gdd.download_file_from_google_drive(file_id='1HmBzuwwRFcjy3_fbOUVKttFMlhuwKxZB',
                                    dest_path=str(datasetPath),
                                    unzip=True)
# delete zip file
datasetPath.unlink()

datasetPath = Path('data/dataset')
maskPath = datasetPath/'with_mask'
nonMaskPath = datasetPath/'without_mask'
randomPath = datasetPath/'random'
maskDF = pd.DataFrame()

for imgPath in tqdm(list(maskPath.iterdir()), desc='with_mask'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 1
    }, ignore_index=True)

for imgPath in tqdm(list(nonMaskPath.iterdir()), desc='without_mask'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 0
    }, ignore_index=True)

for imgPath in tqdm(list(randomPath.iterdir()), desc='random_images'):
    maskDF = maskDF.append({
        'image': str(imgPath),
        'mask': 2
    }, ignore_index=True)

dfName = 'data/dataset/dataset.pickle'
print(f'Saving Dataframe to: {dfName}')
maskDF.to_pickle(dfName)