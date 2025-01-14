import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
import random
from torch.utils.data import Dataset, DataLoader


class TupletDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_negatives=10):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)
        self.class_to_images = {cls: os.listdir(os.path.join(data_dir, cls)) for cls in self.classes}
        self.num_negatives = num_negatives
        print(f"self.num_negatives: {self.num_negatives}")
    
    def __len__(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        # Select the anchor class and image
        anchor_class = self.classes[index]
        anchor_image_path = random.choice(self.class_to_images[anchor_class])
        anchor_image = Image.open(os.path.join(self.data_dir, anchor_class, anchor_image_path)).convert('RGB')

        # Select a positive sample from the same class
        positive_image_path = random.choice(self.class_to_images[anchor_class])
        positive_image = Image.open(os.path.join(self.data_dir, anchor_class, positive_image_path)).convert('RGB')

        # Select negative samples from different classes
        negative_classes = random.sample([cls for cls in self.classes if cls != anchor_class], self.num_negatives)
        negatives = []
        for neg_class in negative_classes:
            neg_image_path = random.choice(self.class_to_images[neg_class])
            negative_image = Image.open(os.path.join(self.data_dir, neg_class, neg_image_path)).convert('RGB')
            negatives.append(self.transform(negative_image) if self.transform else negative_image)
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
        
        negatives = torch.stack(negatives) if negatives else torch.empty(0)
        return anchor_image, positive_image, negatives
