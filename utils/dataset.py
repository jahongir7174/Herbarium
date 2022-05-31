import json
import os
from os.path import *

from PIL import Image
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.transform = transform

        self.samples = self.load_label(data_dir)

    def __getitem__(self, index):
        filename, label = self.samples[index]
        image = self.load_image(filename)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image

    @staticmethod
    def load_label(data_dir):
        cls_names = [folder for folder in os.listdir(data_dir)]
        cls_names.sort()

        cls_to_idx = {cls_name: i for i, cls_name in enumerate(cls_names)}

        images = []
        labels = []

        for root, dirs, filenames in os.walk(data_dir, False, followlinks=True):
            label = basename(relpath(root, data_dir) if (root != data_dir) else '')

            if 0 < len(filenames) <= 80:
                temp = []
                for _ in range(80 // len(filenames)):
                    temp.extend(filenames)

                filenames = temp

            for filename in filenames:
                base, ext = splitext(filename)
                if ext.lower() in ('.png', '.jpg', '.jpeg'):
                    images.append(join(root, filename))
                    labels.append(label)

        return [(i, cls_to_idx[j]) for i, j in zip(images, labels) if j in cls_to_idx]


class TestDataset(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        with open(f'{data_dir}/test.json') as f:
            json_data = json.load(f)
        self.samples = []
        for item in json_data:
            self.samples.append((item['file_name'], item['image_id']))
        cls_names = [folder for folder in os.listdir(f'{data_dir}/train')]
        cls_names.sort()
        self.idx_to_cls = {i: cls_name for i, cls_name in enumerate(cls_names)}

    def __getitem__(self, index):
        filename, image_id = self.samples[index]
        image = self.load_image(f'{self.data_dir}/test/{filename}')
        image = self.transform(image)
        return image, image_id

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(filename):
        with open(filename, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        return image
