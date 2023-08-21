import os 
import pandas as pd 
import torch 
from torch.utils.data import Dataset
from skimage import io

class DrowsyDataset(Dataset):

    def __init__(self, csv_file: str, root_dir: str, transforms=None):
        """Initializes the file paths and transforms for this dataset"""

        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        """Returns the size of this dataset"""

        return len(self.annotations) 
    
    def __getitem__(self, i):
        """Returns a tuple containing the example at index i in the form (image, label)"""

        img_path = os.path.join(self.root_dir, self.annotations.iloc[i, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[i, 1]))

        if self.transforms:
            image = self.transforms(image)

        return (image, y_label)