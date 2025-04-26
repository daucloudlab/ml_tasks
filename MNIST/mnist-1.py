import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transform
from PIL import Image


class DigitDataset(data.Dataset):
    def __init__(self, data_dir, train = True, transform = None):
        self.path = os.path.join(data_dir, 'train' if train else 'test')
        self.transform = transform
        self.targets = torch.eye(10)
        with open(os.path.join(data_dir, 'format.json'), 'r') as f:
            self.format = json.load(f)
        self.length = 0
        self.files = []

        for _dir, _target in self.format.items():
            file_path = os.path.join(self.path, _dir)
            file_lists = os.listdir(file_path)
            self.files.extend(map(lambda x: (os.path.join(file_path, x), _target), file_lists))
            self.length = len(self.files)
        
    def __getitem__(self, index):
        img_path, t = self.files[index]
        img = Image.open(img_path)
        target = self.targets[t]
        if self.transform:
            img = self.transform(img).ravel().float() / 255.0
        return img, target

    def __len__(self):
        return self.length


class DigitNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(inp_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x


to_tensor = transform.ToTensor()
d_train = DigitDataset('dataset', transform = to_tensor)
train_data = data.DataLoader(d_train, batch_size = 32, shuffle = True)

epochs = 4
model = DigitNN(784, 32, 10)
model.train()
optimizer = optim.Adam(params = model.parameters(), lr = 0.01)
loss_func = nn.CrossEntropyLoss()
for _ in range(epochs):
    for x_train, y_train in train_data:
        y_predict = model(x_train)
        loss = loss_func(y_predict, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
d_test = DigitDataset('dataset', train=False, transform=to_tensor)
test_data = data.DataLoader(d_test, batch_size=500, shuffle=False)


for x_test, y_test in test_data:
    with torch.no_grad():
        y_predict = model(x_test)
        y_predict_index = torch.argmax(y_predict, dim=1)
        y_test_index = torch.argmax(y_test, dim=1)
        acc = (y_predict_index == y_test_index).float().mean().item()
        print(f"accuracy = {acc}")
