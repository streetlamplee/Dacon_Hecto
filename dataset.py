import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import torch

class CarCLFDataset(Dataset):
    def __init__(self, data_list:list, img_size:int = 256, transform = None):
        self.data_list = data_list
        self.img_size = img_size
        if transform is None:
            self.transforms_ = transforms.Compose([
                transforms.RandomVerticalFlip(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ColorJitter(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms_ = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        image_path, cls, num_cls= data
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.transforms_(image)
        # image_tensor = image_tensor.permute(2,0,1)

        cls_tensor = torch.tensor([cls], dtype = torch.int32)
        # cls_tensor = F.one_hot(cls_tensor, num_cls)

        return image_tensor, cls_tensor

