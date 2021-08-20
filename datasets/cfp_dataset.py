import torchvision.transforms as T
from torchvision.transforms.functional import resize
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import os
import torch
import numpy as np


class CFPDataset(Dataset):
    """Class representing the CASIA WebFace dataset
    """

    def __init__(self, root_dir, start_index=0, seed=9, transform=None):
        """Init function

    Parameters
    ----------
    data : List[List]
      The path to the data
    transform : :py:class:`torchvision.transforms`
      The transform(s) to apply to the face images

    """
        self.data = []
        id_labels = []
        self.seed = seed

        for root, dirs, files in os.walk(root_dir):
            for name in files:
                path = root.split(os.sep)
                # pose = path[-1]
                # if pose == 'frontal':
                subject = int(path[-1])
                self.data.append(os.path.join(root, name))
                id_labels.append(subject)
        self.id_labels = self.map_labels(id_labels, start_index)
        self.num_classes = len(set(self.id_labels))
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __getitem__(self, idx):
        image_path, class_id = self.data[idx], self.id_labels[idx]

        # dirname, filename = os.path.split(image_path)
        # masked = 0 if filename.split('.')[0].isnumeric() else 1
        image = Image.open(image_path)

        class_id = torch.as_tensor(class_id, dtype=torch.long)
        # masked = torch.as_tensor(masked, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        image = self.to_tensor(image)
        image = resize(image, [128, 128])
        c, h, w = image.size()

        if c < 3:
            image = image.expand(3, -1, -1)

        return image, class_id

    def __len__(self):
        return len(self.data)

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

    def get_data_split(self, test_size):
        """
        Obtain training and validation indices for dataset
        """
        indices = list(range(len(self.data)))
        x_train, _, y_train, _ = train_test_split(self.data, indices,
                                                  test_size=test_size, stratify=self.id_labels,
                                                  random_state=self.seed)

        return x_train, y_train


if __name__ == "__main__":
    SEED = 9

    dataset = CFPDataset(r'../data/CASIA-WebFace', seed=SEED)
    train_ids, val_ids = dataset.get_data_split(0.3)
    train_subset = Subset(dataset, train_ids)
    val_subset = Subset(dataset, val_ids)
