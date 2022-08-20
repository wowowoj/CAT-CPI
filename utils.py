import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, imgs, proteins, label):
        self.imgs = imgs
        self.proteins = proteins
        self.label = label

        self.transformImg = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img_path = self.imgs[item]
        pro_feature = self.proteins[item]
        label_feature = self.label[item]
        img = Image.open(img_path).convert('RGB')
        img = self.transformImg(img)
        print(len(img))
        print(len(pro_feature))
        print(len(label_feature))
        return img, pro_feature, label_feature


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name)]


def data_loader(batch_size, imgs, pro_name, inter_name):
    proteins = load_tensor(pro_name, torch.LongTensor)
    interactions = load_tensor(inter_name, torch.LongTensor)

    dataset = Dataset(imgs, proteins, interactions)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataset_loader


def get_pic_path(pic_path):
    imgs = []
    with open(pic_path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            imgs.append(line.split("\t")[0])
    return imgs
