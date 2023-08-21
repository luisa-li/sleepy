import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import DrowsyDataset
from residual import ResNet18
from residual import Residual
from residual import ResNet
from residual import ResNetSimple

# setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 0.01
batch_size = 32
num_epochs = 100
dropout = 0.2

# loading the data
dataset = DrowsyDataset(csv_file="path.csv", root_dir="data", transforms=transforms.ToTensor())

# splitting data into training, validation and test
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [7873, 1000, 1000])

# setting up data loaders for the model
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# creating the network
model = ResNet18(lr=learning_rate, num_classes=num_classes, dp=dropout).to(device)

# defining optimizer and criterion
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# checking that output size is (32, 2), which is (batch_size, num_classes)
print(model(next(iter(train_loader))[0]).shape)

# initializing the layers with xavier initialization
def init_cnn(module: nn.Module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

model.net.apply(init_cnn)

# training