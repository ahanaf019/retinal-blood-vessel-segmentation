import os
import numpy as np
import cv2 as cv
from glob import glob
from torch import masked_scatter
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, CoarseDropout, OpticalDistortion


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        

def load_data(path):
    # X = images
    # Y = masks
    
    train_x = sorted(glob(os.path.join(path, 'training', 'images', '*.tif')))
    train_y = sorted(glob(os.path.join(path, 'training', '1st_manual', '*.gif')))


    test_x = sorted(glob(os.path.join(path, 'test', 'images', '*.tif')))
    test_y = sorted(glob(os.path.join(path, 'test', '1st_manual', '*.gif')))
    
    return (train_x, train_y), (test_x, test_y)


def augemnt_data(images, mask, save_path, augment=True):
    pass
    


if __name__ == '__main__':
    np.random.seed(42)
    
    # Lead dataset
    data_path = 'DRIVE'
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    
    # Creating directories
    create_dir('new_data/train/image')
    create_dir('new_data/train/mask')
    create_dir('new_data/test/image')
    create_dir('new_data/test/mask')
    
    
    