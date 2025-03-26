import os
import glob
from PIL import Image
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = sorted(os.listdir(self.img_dir), key=lambda x: int(x))
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.classes)
        }

        for label in self.classes:
            for image in glob.glob(os.path.join(self.img_dir, label, "*")):
                self.images.append(image)
                self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(glob.glob(f"{self.img_dir}/*"))
        self.names = [os.path.basename(image)[:-4] for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_name = self.names[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name
