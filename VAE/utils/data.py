import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    def __init__(self, HR_flist, transforms=None):
        super().__init__()
        self.HR_flist = HR_flist
        self.transforms = transforms
    
    def _read_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    
    def __len__(self):
        return len(self.HR_flist)
    
    def __getitem__(self, idx):
        fname = self.HR_flist[idx]
        image = self._read_image(fname)
        image = self.transforms(image=image)['image']
        
        return {"fname": fname, "image": image}

def define_augmentations(size=512):
    train_transform = A.Compose(
        [   
            A.Resize(size, size),
         
            A.OneOf(
                [
                    A.MotionBlur(),
                    A.MedianBlur(blur_limit=3),
                    A.Blur(blur_limit=3),
                ], p=0.2),
        
            A.OneOf([
                A.ChannelShuffle(),
                A.ColorJitter(),
                A.HueSaturationValue(),
            ], p=0.5),
            
            A.Normalize(p=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1),
        ]
    )

    valid_transform = A.Compose(
        [   
            A.Resize(size, size),
         
            A.Normalize(p=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1),
        ]
    )

    return train_transform, valid_transform


def define_datasets_and_dataloaders(
        train_path, valid_path, test_path,
        train_transform, valid_transform,
        batch_size=64, num_workers=16):
    
    train_flist = get_image_flist(train_path)
    valid_flist = get_image_flist(valid_path)
    test_flist = get_image_flist(test_path)
    
    train_dataset = CustomDataset(train_flist, train_transform)
    valid_dataset = CustomDataset(valid_flist, valid_transform)
    test_dataset = CustomDataset(test_flist, valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True, persistent_workers=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, valid_dataloader, test_dataloader


def get_image_flist(path):
    import glob
    flist = sorted(glob.glob(path))
    
    return flist
