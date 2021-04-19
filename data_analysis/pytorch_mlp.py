import torch
import torch.nn as nn
from torch.autograd import Variable
from mlp_utils import load_data
from torch.utils.data import Dataset
import numpy as np

torch.manual_seed(seed=1)

class MyDataset(Dataset):

    def __init__(self, data, labels=None, is_test=False):
        self.data = data
        self.labels = np.array([labels])
        self.is_test = is_test

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.data[idx, :]
        if not self.is_test:
            y = self.labels[:, idx]
        else:
            y = []

        return X, y


import sys
data_name = sys.argv[1]
num_epochs = int(sys.argv[2])
# Hyper Parameters
input_size = 256
hidden_size = 128
num_classes = 1
batch_size = 22
learning_rate = 0.001

X_train, y_train, X_test, emotions = load_data(data_name)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=MyDataset(X_train, y_train),
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=MyDataset(X_test, is_test=True),
                                          batch_size=batch_size,
                                          shuffle=False)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        out = self.fc1(x)
        out = self.relu(out)
        #out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.tanh(out)
        return out

net = Net(input_size, hidden_size, num_classes)
#net.cuda()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train the Model
prev_loss = 9999
flag = 0
for epoch in range(num_epochs):
    for i, (Xs, ys) in enumerate(train_loader):
        # Convert torch tensor to Variable
        Xs = Variable(Xs).float()
        ys = Variable(ys).float()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(Xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()

        cur_loss = loss.data.item()
        if prev_loss - cur_loss < 0.0001:
            print('Early Stop: Epoch [{}/{}], Step [{}], Loss: {}'.format(epoch+1, num_epochs, i+1, loss.data))
            flag = 1
            break
        prev_loss = cur_loss
        print ('Epoch [{}/{}], Step [{}], Loss: {}'.format(epoch+1, num_epochs, i+1, loss.data))
    if flag:
        break

# Test the Model
for Xs, _ in test_loader:
    Xs = Variable(Xs).float()
    outputs = net(Xs)
    for i, (output) in enumerate(outputs):
        print('{0}\t{1:.2f}'.format(emotions[i], output.data[0].item()))


# Save the Model
torch.save(net.state_dict(), 'model.pkl')
