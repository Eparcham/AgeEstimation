
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
                'Image Name': name,
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

df_train, df_temp = train_test_split(df, test_size=0.3, stratify = df.Age)
df_test, df_valid = train_test_split(df_temp, test_size=0.5, stratify = df_temp.Age)
print(df_train.shape, df_test.shape)

df_train.to_csv("train_data.csv", index=False)
df_test.to_csv("test_data.csv", index=False)
df_valid.to_csv("valid_data.csv", index=False)