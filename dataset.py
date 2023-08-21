import os 
import pandas as pd 
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
        pass