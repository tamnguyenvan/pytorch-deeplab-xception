import os
import torch
import argparse

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from dataloaders.utils import decode_seg_map_sequence

from modeling.deeplab import *


input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def load_image(filename, size=None, scale=None, keep_asp=True, transform=input_transform):
    """Load the image for demos"""
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)

    if transform:
        img = transform(img)
    return img


def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.model, map_location=device)
    model = DeepLab(num_classes=14,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # Load image
    img = load_image(args.source).to(device).unsqueeze(0)
    
    # Make prediction
    output = model(img)
    predict = torch.max(output, 1)[1].cpu().numpy()

    floor_mask = np.squeeze(predict)

    img_origin = cv2.imread(args.source)
    h, w = img_origin.shape[:2]
    floor_img = cv2.resize(cv2.imread(args.floor), (w, h))
    img_origin[floor_mask == 5] = 0
    floor_img[floor_mask != 5] = 0
    output = img_origin + floor_img
    cv2.imwrite('output.png', output)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DeepLabv3+ demo')
    parser.add_argument('--source', type=str, help='Path to image source')
    parser.add_argument('--floor', type=str, help='Path to floor image.')
    parser.add_argument('--model', type=str, help='Path to saved model file')
    args = parser.parse_args()
    print(args)

    main()
