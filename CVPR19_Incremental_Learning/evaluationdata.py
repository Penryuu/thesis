import os
from PIL import Image
import torch.utils.data as data
import cv2
from typing import Any, Callable, Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

class CustomDataset(data.Dataset):
    @staticmethod
    def get_label(img_path):
        # Extract the label from the filename
        filename = os.path.basename(img_path)
        label = filename.split("_")[0]
        return label

    def __init__(self, root: str, transform: Optional[Callable] = None,target_transform: Optional[Callable] = None) -> None:
        # super().__init__(root, transform=transform, target_transform=target_transform), train: bool = True
        self.root = root
        self.transform = transform
        # self.train = train
        self.target_transform = target_transform
        self.data: Any = []
        self.targets = []
         # now load the picked numpy arrays
        for root, dirs, files in os.walk(root):
            for file in files:
                if file.endswith(".png"):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (32, 32))
                    self.data.append(img)
                    self.targets.append(CustomDataset.get_label(img_path))

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        label_encoder = LabelEncoder()
        self.targets = label_encoder.fit_transform(self.targets)
        # if self.train:
        #     self.targets = label_encoder.fit_transform(self.targets)
        # else:
        #     self.targets = label_encoder.transform(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# from torchvision import datasets
# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# # # Write transform for image
# data_transform = transforms.Compose([
#     # Resize the images to 64x64
#     transforms.Resize(size=(32, 32)),
#     # Flip the images randomly on the horizontal
#     transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
#     # Turn the image into a torch.Tensor
#     transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
# ])
# train_dir = 'Z:\ProposalAlgos\humandetectiondataset\\train1'
# train_data = CustomDataset(train_dir,transform=data_transform,target_transform=None) # transforms to perform on labels (if necessary)
# test_dir = 'Z:\ProposalAlgos\humandetectiondataset\\test1'
# test_data = CustomDataset(test_dir,transform=data_transform)


# print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
# print('train_data.data: ', train_data.data)
# print('train_data.targets: ', train_data.targets)