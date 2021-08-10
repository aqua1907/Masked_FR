import torch
import os
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import matplotlib.pyplot as plt


class CasiaWebFaceDataset(Dataset):
    """Class representing the CASIA WebFace dataset
  """

    def __init__(self, images, labels, transform=None):
        """Init function

    Parameters
    ----------
    data : List[List]
      The path to the data
    transform : :py:class:`torchvision.transforms`
      The transform(s) to apply to the face images

    """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx):
        image_path, class_id = self.images[idx], self.labels[idx]

        dirname, filename = os.path.split(image_path)
        masked = 0 if filename.split('.')[0].isnumeric() else 1
        image = Image.open(image_path)

        class_id = torch.as_tensor(class_id, dtype=torch.long)
        masked = torch.as_tensor(masked, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        image = self.to_tensor(image)
        image = resize(image, [128, 128])
        image = image.expand(3, -1, -1)

        return image, class_id, masked

    def __len__(self):
        return len(self.images)


class MyLoader:
    def __init__(self, root_dir, start_index=0, batch_size=32, test_size=0.25, seed=123):

        self.data = []
        id_labels = []
        self.seed = seed

        for root, dirs, files in os.walk(root_dir):
            for name in files:
                path = root.split(os.sep)
                subject = int(path[-1])
                self.data.append(os.path.join(root, name))
                id_labels.append(subject)

        self.id_labels = self.map_labels(id_labels, start_index)
        self.num_classes = len(set(self.id_labels))

        self.test_size = test_size
        self.batch_size = batch_size

        # Create new subsets according to test_size ratio
        x_train, x_val, y_train, y_val = self.get_data_split()  # obtain indices for each subset

        self.train_dataset = CasiaWebFaceDataset(x_train, y_train, self.get_transforms())
        self.val_dataset = CasiaWebFaceDataset(x_val, y_val)


    @staticmethod
    def map_labels(raw_labels, start_index=0):
        """
      Map the ID label to [0 - # of IDs]

      Parameters
      ----------
      raw_labels: list of :obj:`int`
        The labels of the samples

      """
        possible_labels = sorted(list(set(raw_labels)))
        labels = np.array(raw_labels)

        for i in range(len(possible_labels)):
            l = possible_labels[i]
            labels[np.where(labels == l)[0]] = i + start_index

        return labels

    @staticmethod
    def labels_weights(classes):
        """
        Compute weights for each class
        :param (list) classes: list of classes
        :return: weight value for each class
        """
        counts = np.bincount(classes)
        labels_weights = 1. / counts
        weights = labels_weights[classes]

        return weights

    def get_data_split(self):
        """
        Obtain training and validation indices for dataset
        """
        x_train, x_val, y_train, y_val = train_test_split(self.data, self.id_labels,
                                                          test_size=self.test_size, stratify=self.id_labels,
                                                          random_state=self.seed)

        return x_train, x_val, y_train, y_val

    def create_loaders(self):
        # weights = self.labels_weights(self.train_dataset.labels)
        # train_sampler = WeightedRandomSampler(weights, len(self.train_dataset), replacement=True)
        train_loader = DataLoader(self.train_dataset, self.batch_size,
                                  pin_memory=True, drop_last=True)

        val_loader = DataLoader(self.val_dataset, self.batch_size,
                                shuffle=True, pin_memory=True)

        return train_loader, val_loader

    @staticmethod
    def get_transforms():
        """
        Specify the list of augmentations and using torchvision RandomChoice
        to select randomly one of them on the next iteration of the dataloader
        :return: list of transformations
        """
        transforms = T.Compose([T.RandomHorizontalFlip(p=0.35),
                                T.RandomVerticalFlip(p=0.35),
                                T.RandomPerspective(p=0.35),
                                ])

        return transforms


if __name__ == "__main__":
    myloader = MyLoader(r'data/CASIA-WebFace', batch_size=32, test_size=0.3, seed=1364)
    train_loader, val_loader = myloader.create_loaders()
    train_dataset, val_dataset = myloader.train_dataset, myloader.val_dataset
    print(len(myloader.id_labels))

    # for i in tqdm(range(len(train_dataset))):
    #     img, image_path, _, _ = train_dataset[i]
    #     c, h, w = img.size()
    #
    #     if c < 3:
    #         print(image_path)
    # print()
    # for i in tqdm(range(len(val_dataset))):
    #     img, image_path, _, _ = val_dataset[i]
    #     c, h, w = img.size()
    #
    #     if c < 3:
    #         print(image_path)
    # train_loader, val_loader = myloader.create_loaders()
    #
    imgs, _, _ = next(iter(train_loader))
    print(imgs.shape)
    grid = make_grid(imgs).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()
