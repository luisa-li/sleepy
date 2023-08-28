import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import DrowsyDataset
from residual import ResNet18
from tqdm import tqdm, trange
import time

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
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [7871, 1000, 1000])

# setting up data loaders for the model
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# creating the network
model = ResNet18(lr=learning_rate, num_classes=num_classes, dp=dropout).to(device)

# defining optimizer and criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# checking that output size is (32, 2), which is (batch_size, num_classes)
print(model(next(iter(train_loader))[0]).shape)

# initializing the layers with xavier initialization
def init_cnn(module: nn.Module):
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

model.net.apply(init_cnn)

# methods for the training loop 
def calculate_accuracy(y_pred, y):

    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

def train():

    iterator = train_loader

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=False):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate():

    iterator = valid_loader

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for (x, y) in tqdm(iterator, desc="Evaluating", leave=False):

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

# training loop 
for epoch in trange(num_epochs, desc="Epochs"):

    start_time = time.monotonic()

    train_loss, train_acc = train()
    valid_loss, valid_acc = evaluate()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'DROWSY.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')