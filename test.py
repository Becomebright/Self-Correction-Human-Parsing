import os
import torch
import argparse
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2

from model import network
from datasets import SCHPDataset, transform_logits


model = network(num_classes=20, pretrained=None)
model.load_state_dict(torch.load('HumanParsing/model/exp-schp-201908261155-lip.pth'))
model.eval()
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
im = cv2.imread('input/demo.jpg')

