from sqlite3 import DatabaseError
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import Tensor
import numpy as np
#device = torch.device("mps")
device = torch.device("cpu")

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, bias=False)
        #self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2304, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

model = MnistModel()
model = model.to(device)

batch_size = 400
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000)


optimizer = optim.Adam(model.parameters(), lr=0.0001)

model.train()
i = 0
for epoch in range(100):
    for data, target in train_loader:
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        if i % 100 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.data, accuracy))
        i += 1
print('Total accuracy :',accuracy)
model = model.to(torch.device("cpu"))
torch.save(model.state_dict(),'model.pth')


## simple code to change pth to npy..
torchdict = torch.load('model.pth')
numpydict = {}
numpydict['conv1w'] = np.array(torchdict['conv1.weight'])

numpydict['conv2w'] = np.array(torchdict['conv2.weight'])

numpydict['fc3w'] = np.array(torchdict['fc1.weight'])


np.save('model.npy', numpydict, allow_pickle=True)