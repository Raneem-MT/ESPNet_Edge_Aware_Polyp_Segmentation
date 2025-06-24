from torch.utils.data import Dataset
import torch
import numpy as np
from imageio import imwrite, imread
import os
import torch.nn.functional as F
#import albumentations as A
import cv2
from torchvision import transforms
import random


#new std [0.304, 0.217, 0.176]
def img_normalize(image):

    image = (image-np.array([0.538, 0.312, 0.237], dtype=np.float32).reshape((1, 1, 3)))\
        /np.array([0.260, 0.190, 0.153], dtype=np.float32).reshape((1, 1, 3))
    return image
    

class TrainDataset(Dataset):
    def __init__(self, paths):
        self.image = []
        self.label = []
        self.edge = []  # To store edge paths
        self.count = {}
        for path in paths:
            self.list = os.listdir(os.path.join(path, "images"))
            for i in self.list:
                self.image.append(os.path.join(path, "images", i))
                self.label.append(os.path.join(path, "masks", i.split(".")[0] + ".png"))
                self.edge.append(os.path.join(path, "edges", i.split(".")[0] + ".png"))  # Add edge path
        print("Datasetsize:", len(self.image))
        
    def __len__(self):
        return len(self.image)
        
    def __getitem__(self, item):
        img = imread(self.image[item]).astype(np.float32) / 255.
        label = imread(self.label[item]).astype(np.float32) / 255.
        edge = imread(self.edge[item]).astype(np.float32) / 255.  
       
        #print(f"Img shape: {img.shape}, Label shape: {label.shape}, Edge shape: {edge.shape}")

        # Apply random augmentations
        ratio = np.random.rand()
        if ratio < 0.25:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
            edge = cv2.flip(edge, 1)  # Flip the edge image
        elif ratio < 0.5:
            img = cv2.flip(img, 0)
            label = cv2.flip(label, 0)
            edge = cv2.flip(edge, 0)  # Flip the edge image
        elif ratio < 0.75:
            img = cv2.flip(img, -1)
            label = cv2.flip(label, -1)
            edge = cv2.flip(edge, -1)  # Flip the edge image

        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = label[:, :, np.newaxis]
        
        if len(edge.shape)==3:
            edge=edge[:,:,0]       
        edge=edge[:,:,np.newaxis]
        
        return {"img": torch.from_numpy(img_normalize(img)).permute(2,0,1).unsqueeze(0),
                "label":torch.from_numpy(label).permute(2,0,1).unsqueeze(0),
                "edge":torch.from_numpy(edge).permute(2,0,1).unsqueeze(0)}


def my_collate_fn(batch):
    size = 384
    imgs=[]
    labels=[]
    edges=[]
    for item in batch:
        imgs.append(F.interpolate(item['img'], (size, size), mode='bilinear'))
        labels.append(F.interpolate(item['label'], (size, size), mode='bilinear'))
        edges.append(F.interpolate(item['edge'], (size, size), mode='bilinear'))
    return {'img': torch.cat(imgs, 0),
            'label': torch.cat(labels, 0),
            'edge': torch.cat(edges, 0)}
##########################################################################################
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
Dirs = [
        "/resstore/b0211/Data/polypData/SUN-SEG/TestHardDataset/Seen/",
    ]
for dataset_dir in Dirs:
        dataset_name = "/".join(dataset_dir.rstrip("/").split("/")[-2:])
        print(f"Processing dataset: {dataset_name}")

        # Load dataset
        Dataset = TestSUN([dataset_dir], size=384)
        Dataloader = DataLoader(Dataset, batch_size=1, num_workers= 2)
        for data in tqdm(Dataloader):
            img = data['img']
            print(f"Image shape: {img.shape}")
'''


class TestDataset(Dataset):
    def __init__(self, path, size):
        self.size = size
        self.image = []
        self.label = []
        self.list = os.listdir(os.path.join(path, "images"))
        self.count = {}
        for i in self.list:
            self.image.append(os.path.join(path, "images", i))
            self.label.append(os.path.join(path, "masks", i.split(".")[0] + ".png"))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        img = imread(self.image[item]).astype(np.float32) / 255.
        label = imread(self.label[item]).astype(np.float32) / 255.
        
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        
        return {
            "img": F.interpolate(torch.from_numpy(img_normalize(img)).permute(2, 0, 1).unsqueeze(0), 
                                 (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0),
            "label": torch.from_numpy(label).permute(2, 0, 1),
            'name': self.label[item]
        }




class TestSUN(Dataset):

    def __init__(self, paths, size):  
        self.size = size
        self.image = []
        self.label = []

        for path in paths:
            frame_path = os.path.join(path, "Frame")
            gt_path = os.path.join(path, "GT")

            for case in os.listdir(frame_path):
                case_frame_path = os.path.join(frame_path, case)
                case_gt_path = os.path.join(gt_path, case)

                if not os.path.isdir(case_frame_path) or not os.path.isdir(case_gt_path):
                    continue

                for img_file in os.listdir(case_frame_path):
                    img_path = os.path.join(case_frame_path, img_file)
                    mask_path = os.path.join(case_gt_path, img_file.split(".")[0] + ".png")

                    if os.path.exists(mask_path):  # Ensure the mask exists
                        self.image.append(img_path)
                        self.label.append(mask_path)

        print("Datasetsize:", len(self.image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):
        img = imread(self.image[item]).astype(np.float32) / 255.
        label = imread(self.label[item]).astype(np.float32) / 255.
        if len(label.shape) == 2:
            label = label[:, :, np.newaxis]
        return {"img": F.interpolate(torch.from_numpy(img_normalize(img)).permute(2, 0, 1).unsqueeze(0), 
                                     (self.size, self.size), mode='bilinear', align_corners=True).squeeze(0),
                "label": torch.from_numpy(label).permute(2, 0, 1),
                'name': self.label[item]}
           