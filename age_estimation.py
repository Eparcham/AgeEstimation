## Problem Statement:
# Age Detection from Facial Features
# The objective of this project is to develop a robust and
# efficient algorithm for age detection based on facial features.
# With the rapid advancement in machine learning and computer vision,
# there is a growing interest in understanding demographic attributes from facial images.
# Age detection, particularly, has significant applications
# in areas such as security, marketing, and human-computer interaction.
# Our approach should involve the utilization of deep learning techniques
# to analyze facial characteristics that are indicative of age.
# This includes, but is not limited to, wrinkles, facial structure,
# and skin texture. The algorithm should be trained on a diverse dataset containing a wide range of ages,
# ethnicities, and lighting conditions to ensure accuracy and reduce bias.
# Furthermore, the system should be able to handle real-time processing,
# providing quick and accurate age estimations. It must also respect privacy
# and ethical guidelines, ensuring that the data is used solely for the intended
# purpose without infringing on individual privacy rights.
# In summary, the challenge lies in creating a model that is not only precise in age estimation
# but also fast, reliable, and ethical, adapting to the varying complexities of human faces while being sensitive to privacy concerns.

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import PIL
from PIL import Image
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import tqdm
import torchmetrics as tm
import glob

## Dataset

Dir_Dataset = "H:/PytorchClass/Data/UTKFace/utkface_aligned_cropped/UTKFace/"
list_image = glob.glob(os.path.join(Dir_Dataset, "*.jpg"))
Identity = ["White", "Black", "Asian", "Indian", "Others"]
data = []
for full_path in list_image:
    name = os.path.basename(full_path)
    split_name = name.split("_")
    if len(split_name) >= 4 and split_name[2].isdigit():
        age = int(split_name[0])
        if age < 80:
            gender = "Male" if split_name[1] == '0' else "Female"
            id_ = Identity[int(split_name[2])]
            data.append({
                'ImageName': name,
                'ID': id_,
                'Gender': gender,
                'Age': age
            })
    else:
        print(name)

df = pd.DataFrame(data)

file_path = 'UTKFace.csv'
df.to_csv(file_path, index=False)

print(f"Dataframe saved to {file_path}")

## split datset

df_train, df_temp = train_test_split(df, test_size=0.3, stratify=df.Age)
df_test, df_valid = train_test_split(df_temp, test_size=0.5, stratify=df_temp.Age)
print(df_train.shape, df_test.shape)

df_train.to_csv("train_data.csv", index=False)
df_test.to_csv("test_data.csv", index=False)
df_valid.to_csv("valid_data.csv", index=False)

## train and test transform
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


## costom datset
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, file_csv, transform):
        self.root_dir = root_dir
        self.csv_file = file_csv
        self.transform = transform
        self.data = pd.read_csv(self.csv_file)
        self.gender_dict = {'Male': 0, 'Female': 1}
        self.id_dict = {"White": 0, "Black": 1, "Asian": 2, "Indian": 3, "Others": 4}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data.iloc[item, :]

        image_name = sample.ImageName
        age = sample.Age
        gender = sample.Gender
        country = sample.ID

        img = Image.open(self.root_dir + image_name)
        img = self.transform(img)

        age = torch.tensor([age], dtype=torch.float32)  ## very important to use one class use [] in input

        country = torch.tensor(self.id_dict[country], dtype=torch.float32)
        gender = torch.tensor(self.gender_dict[gender], dtype=torch.float32)

        return img, age, country, gender


# test dataset
temp_datset = UTKFaceDataset(root_dir=Dir_Dataset, file_csv="train_data.csv", transform=train_transform)
img, age, country, gender = temp_datset[1]
a = 1
## create loader

train_set = UTKFaceDataset(root_dir=Dir_Dataset, file_csv="train_data.csv", transform=train_transform)
test_set = UTKFaceDataset(root_dir=Dir_Dataset, file_csv="test_data.csv", transform=test_transform)
valid_set = UTKFaceDataset(root_dir=Dir_Dataset, file_csv="valid_data.csv", transform=test_transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)


## Model

class ModelAGEDecetion(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)

    def forward(self, x):
        return self.model(x)


def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


self_ = ModelAGEDecetion()
print(self_.model)

## config

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = self_.model.to(device)
print(f"num trainable params: {num_trainable_params(model)}")

loss_fn = nn.L1Loss()

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)

metric = tm.MeanAbsoluteError().to(device)
# metric = tm.Accuracy().to(device)
 
def train_one_epoch(train_loader, model, loss_fn, device, optimizer, metric, epoch=None):
    model = model.train()
    loss_train = AverageMeter()
    metric.reset()
    with tqdm.tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets, _, _ in tepoch:
            if epoch:
                tepoch.set_description(f"Epoch {epoch}")
            targets = targets.to(device)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loss_train.update(loss.item(), n=len(targets))
            # outputs = (outputs > 0.5).int()
            metric.update(outputs, targets)
            tepoch.set_postfix(loss=loss_train.avg, metric=metric.compute().item())

    return model, loss_train.avg, metric.compute().item()


train_one_epoch(train_loader, model, loss_fn, device, optimizer, metric, epoch=1)
