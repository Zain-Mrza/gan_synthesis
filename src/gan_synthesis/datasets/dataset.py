import torch
from torch.utils.data import Dataset, random_split

from gan_synthesis.preprocessing.transforms import read_cropped


class Dataset(Dataset):  # file names are indexed at 1
    def __init__(self):
        self.weights = torch.tensor([0.1614, 0.2952, 0.2217, 0.3217])

    def __len__(self):
        return 369

    def __getitem__(self, idx):
        image = read_cropped(idx, "contrast")
        mask = read_cropped(idx, "seg")

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.int32).unsqueeze(0)

        return image, mask

    def split(self, trainp=0.8):
        if trainp > 1:
            raise ValueError(
                "Parameter must be percent (as a decimal) of dataset to be train set. Must be less than 1."
            )
        train_size = int(trainp * len(self))
        test_size = len(self) - train_size

        generator = torch.Generator().manual_seed(42)

        return random_split(self, [train_size, test_size], generator=generator)
        # returns train_dataset, test_dataset (still need to instantiate withe DataLoader)
