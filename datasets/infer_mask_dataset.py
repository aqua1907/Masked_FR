import torchvision.transforms as T
from torchvision.transforms.functional import resize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch


class InferMask(Dataset):
    def __init__(self, unmasked_dir, masked_dir):
        super(InferMask, self).__init__()

        self.unmasked = self._gather_imgs(unmasked_dir)
        self.masked = self._gather_imgs(masked_dir)
        self.data = self.unmasked + self.masked

        self.to_tensor = T.ToTensor()

    @staticmethod
    def _gather_imgs(path):
        data = []
        for root, dirs, files in os.walk(path):
            for name in files:
                data.append(os.path.join(root, name))

        return data

    def __getitem__(self, idx):
        img_path = self.data[idx]
        path_split = img_path.split(os.path.sep)[1]

        img = Image.open(img_path)
        img = self.to_tensor(img)
        img = resize(img, [128, 128])

        label = 0 if path_split == 'unmasked' else 1
        label = torch.as_tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.data)
