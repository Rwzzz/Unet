
# ================================================================== #
#                Input pipeline for custom dataset                 #
# ================================================================== #

# You should build your custom dataset as below.
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Initialize file paths or a list of file names.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         pass
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0
#
# # You can then use the prebuilt data loader.
# custom_dataset = CustomDataset()
# train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
#                                            batch_size=64,
#                                            shuffle=True)




import torch

import cv2

import os

import glob

from torch.utils.data import Dataset

import random

class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))

    def augment(self, image, flipcode):
        flip = cv2.flip(image, flipcode)
        return flip

    def __getitem__(self, index):
        #读取图片和标签
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')
        image = cv2.imread(image_path) #RGB 3通道图片
        label = cv2.imread(label_path)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        #label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        #对图片进行预处理preprocess
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label/255
        #图像增强
        flipcode = random.choice([-1, 0, 1, 2])
        if(flipcode != 2):
            image = self.augment(image, flipcode)
            label = self.augment(label, flipcode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    data_path = "F:\labelme\\train_image\\"
    plate_dataset = SelfDataSet(data_path)
    print(len(plate_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=plate_dataset, batch_size=5, shuffle=True)
    for image,label in train_loader:
        print(label.shape)