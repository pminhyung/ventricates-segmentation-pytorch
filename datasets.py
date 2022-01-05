from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from PIL import Image

from glob import glob

class CustomDataset(Dataset):
    def __init__(self, df, mode = 'train', transform = None, has_masks:bool=True):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.df = df

        # 정답파일 존재 Flag (True 시 Score 계산 목적)
        self.has_masks = has_masks
        
    def __getitem__(self, idx: int):
        imgfname = self.df.loc[idx, 'filename']
        imgsize = self.df.loc[idx, 'size']
        image = cv2.imread(self.df.loc[idx, 'filename'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        maskfname = imgfname.replace('png','npy')
        if self.has_masks:
            mask = np.load(maskfname).astype(np.int8)
        
        if self.mode=='train':
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            return image, mask, imgfname
        
        if self.mode=='val':
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]
            return image, mask, imgfname, imgsize

        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=image)
                images = transformed["image"]
            if self.has_masks:
                return images, mask, imgfname, imgsize
            else:
                return images, imgfname, imgsize
    
    def __len__(self) -> int:
        return len(self.df)

def make_data_df(dir_path):
    df = pd.DataFrame({'filename':list(glob(dir_path+'/*.png')),})
    df['size'] = df['filename'].apply(lambda path : Image.open(path).size)
    return df

def make_mask_df(dir_path):
    df = pd.DataFrame({'filename':list(glob(dir_path+'/*.npy')),})
    return df

def collate_fn(batch):
    return tuple(zip(*batch))

class ScoreDataset(Dataset):
    def __init__(self, mask_df, pred_df):
        super().__init__()
        self.mask_df = mask_df
        self.pred_df = pred_df
        
    def __getitem__(self, idx: int):
        mask_path = self.mask_df.loc[idx, 'filename']
        pred_path = self.pred_df.loc[idx, 'filename']

        mask = np.load(mask_path).astype(np.int8)
        pred = np.load(pred_path).astype(np.int8)
        
        return mask, pred
    
    def __len__(self) -> int:
        return len(self.mask_df)