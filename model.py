#%%
import torch

print("cuda.is_available :",torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

#%%
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        #2854 * 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(150,5))
        #2705 * 124
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(306,5))
        #2400 * 120
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(601,21))
        #1800 * 100
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1201,5))
        #600 * 96

        #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(600 * 96 *32, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dp3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(64, 48)
        self.bn4 = nn.BatchNorm1d(48)

        self.fc5 = nn.Linear(48, 10)

        print("model structure compelete!")

    def forward(self, x):

        print("state : conv_1 (2854 * 128 -> 2705 * 124)",end="\r")
        x = func.relu(self.conv1(x))

        print("state : conv_2 (2705 * 124 -> 2400 * 120)",end="\r")
        x = func.relu(self.conv2(x))

        print("state : conv_3 (2400 * 120 -> 1800 * 100)",end="\r")
        x = func.relu(self.conv3(x))

        print("state : conv_4 (1800 * 100 -> 600 * 96)",end="\r")
        x = func.relu(self.conv4(x))
        
        print("state : view (600 * 96)",end="\r")
        x = x.view(-1, 600 * 96 *32)

        print("state : fc_1 (600 * 96 * 32 -> 512)",end="\r")
        x = func.relu(self.bn1(self.fc1(x)))

        print("state : dp_1 (512)",end="\r")
        x = self.dp1(x)

        print("state : fc_2 (512 -> 128)",end="\r")
        x = func.relu(self.bn2(self.fc2(x)))

        print("state : fc_3 (128 -> 64)",end="\r")
        x = func.relu(self.bn3(self.fc3(x)))

        print("state : dp_3 (64)",end="\r")
        x = self.dp3(x)

        print("state : fc_4 (64 -> 48)",end="\r")
        x = func.relu(self.bn4(self.fc4(x)))
        
        print("state : softmax (64 -> 10)",end="\r")
        x = self.fc5(x)
        x = func.softmax(x, dim=1)

        print("state : done",end="\r")
        return x

#%%
import math
import matplotlib.pyplot as plt

def plt_save(lossA=[], lossB=[], accA=[], epoch_count=1):

    # accuracy plt
    plt.subplot(1,2,1)
    plt.xlim((0,epochs/epoch_count))
    plt.ylim((0,100))
    plt.plot(accA, "b-", lw=1)

    plt.xlabel("epochs")
    plt.ylabel("accuracy rate (%)")

    plt.axhline(y=60, c="m", ls="--", lw=0.5)
    plt.axhline(y=70, c="y", ls="--", lw=0.5)

    # loss plt
    plt.subplot(1,2,2)
    plt.xlim((0,epochs/epoch_count))
    plt.ylim((0,math.log2(10)))
    lossA = lossA
    lossB = lossB
    plt.plot(lossA, "r-", lossB, "b-", lw=1)

    plt.xlabel("epochs")
    plt.ylabel("Loss")

    plt.axhline(0.5, c="m", ls="--", lw=0.5)
    plt.axhline(1.5, c="m", ls="--", lw=0.5)

    plt.legend(('training','valid'),loc=1)

    plt.tight_layout()

#%%
"""[train dataset is here](https://drive.google.com/drive/folders/1-0ilLOm9mtlTZVM6QQC5FZaW7KtEQ6Mg?usp=sharing)"""

import numpy as np
from tqdm.notebook import trange

x_path = r"data\x_data.npy"
y_path = r"data\y_data.npy"

if torch.cuda.is_available():
    x_tensor = torch.from_numpy(np.load(x_path)).float().to(device)
    y_tensor = torch.from_numpy(np.load(y_path)).float().to(device)
else:
    x_tensor = torch.from_numpy(np.load(x_path)).float()
    y_tensor = torch.from_numpy(np.load(y_path)).float()
train_size = int(0.8*len(x_tensor))
val_size = (len(x_tensor) - train_size)
train_data, val_data = random_split(TensorDataset(x_tensor,y_tensor),[train_size,val_size])

train_loader = DataLoader(train_data,40,shuffle=True)
val_loader = DataLoader(val_data,40,shuffle=True)
print("data loading compelete!")

#%%
if torch.cuda.is_available():
    model = CNNModel().to(device)
else:
    model = CNNModel()

epochs = 1000

crit = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=5e-3)

acc_his = []
val_loss_his = []
train_loss_his = []

model.eval()
if torch.cuda.is_available():
    x = torch.tensor([np.random.randn(2854,128)]).float().to(device)
else: 
    x = torch.tensor([np.random.randn(2854,128)]).float()
model(x)
print("model testing compelete!")

#%%
for epoch in trange(epochs):

    train_loss = 0
    model.train()

    for x_batch, y_batch in train_loader:

        opt.zero_grad()
        y_pred = model(x_batch)

        loss = crit(y_pred, y_batch)
        loss.backward()

        opt.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    if (epoch+1)%25 == 0:
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                loss = crit(y_pred, y_batch)
                val_loss += loss.item()

                preds = torch.argmax(y_pred, dim=1)
                y_correct = torch.argmax(y_batch,dim=1)
                correct += (preds == y_correct).sum().item()
                total += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc = correct/total
        print(f"epoch : {epoch+1: >3}\033[0m/{epochs: >3}\033[0m   ",
                f"train_loss : {train_loss: >.8f}\033[0m   ",
                f"val_loss :   {val_loss: >.8f}\033[0m   ",
                f"val_acc : {100*val_acc: > 3.2f}%\033[0m")

        acc_his.append(100*val_acc)
        val_loss_his.append(val_loss)
        train_loss_his.append(train_loss)

#%%
plt.figure(figsize=(7,3))
plt_save(train_loss_his, val_loss_his, acc_his,25)
plt.savefig(r"model2_training\training_plot.png")
plt.show()
torch.save(model.state_dict(),r"model2_training\model_parms.pth")