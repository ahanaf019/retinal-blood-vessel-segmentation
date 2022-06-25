import os
import numpy as np
import cv2 as cv
from glob import glob
from torch import masked_scatter, prelu
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


def augemnt_data(images, masks, save_path, augment=True):
    # Size
    H = 512
    W = 512
    
    for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extracting names

        name = x.split("\\")[-1].split('.')[0]
        
        # Reading image and mask
        x = cv.imread(x, cv.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        
        
        if augment is True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]
            
            aug =GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]
            
            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]
            
            
            X = [x, x1, x2, x3, x4, x5]
            Y = [y, y1, y2, y3, y4 ,y5] 
            
            
        else:
            X = [x]
            Y = [y]

        # i = image, m = mask 
        index = 0
        for i, m in zip(X, Y):
            cv.resize(i, (W, H))
            cv.resize(m, (W, H))
            
            if len(X) == 1:
                tmp_image_name = f"{name}.bmp"
                tmp_mask_name = f"{name}.bmp"
            else:
                tmp_image_name = f"{name}_{index}.bmp"
                tmp_mask_name = f"{name}_{index}.bmp"
            
            image_path = os.path.join(save_path, 'image', tmp_image_name)
            mask_path = os.path.join(save_path, 'mask', tmp_mask_name)
            
            cv.imwrite(image_path, i)
            cv.imwrite(mask_path, m)
            
            index += 1

    


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
    
    augemnt_data(train_x, train_y, 'new_data/train/', augment=True)
    augemnt_data(test_x, test_y, 'new_data/test/', augment=False)
    