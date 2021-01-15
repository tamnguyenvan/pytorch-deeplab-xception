from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class NYUv2Segmentation(Dataset):
    """
    NYUv2 Segmentation dataset
    """
    NUM_CLASSES = 14

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('nyuv2'),
                 split='train',
                 ):
        """
        :param base_dir: path to NYUv2 dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images')
        self._cat_dir = os.path.join(self._base_dir, 'annotations')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.im_ids = []
        self.images = []
        self.categories = []

        if split == 'train':
            image_dir = os.path.join(self._image_dir, 'training')
            cat_dir = os.path.join(self._cat_dir, 'training')
        else:
            image_dir = os.path.join(self._image_dir, 'validation')
            cat_dir = os.path.join(self._cat_dir, 'validation')
        
        img_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        for img_path in img_paths:
            _image = img_path
            filename = os.path.splitext(os.path.basename(img_path))[0].split('_')[-1]
            _cat = os.path.join(cat_dir, 'new_nyu_class13_' + filename + '.png')
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'NYUv2(split=' + str(self.split) + ')'
