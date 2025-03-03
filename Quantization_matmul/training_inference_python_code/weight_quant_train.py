import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np

train_loader = DataLoader(datasets.MNIST('././data', train=True, download=True, transform=transforms.ToTensor()), batch_size=50, shuffle=True)
test_loader = DataLoader(datasets.MNIST('././data', train=False, transform=transforms.ToTensor()), batch_size=1000)

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

model = MnistModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
batch_size = 50
i = 0
for epoch in range(3):
    for data, target in train_loader:
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        prediction = output.data.max(1)[1]
        accuracy = prediction.eq(target.data).sum().item() / batch_size * 100

        if i % 100 == 0:
            print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(i, loss.item(), accuracy))
        i += 1 # each one batch

def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            total_correct += prediction.eq(target.view_as(prediction)).sum().item()
    return total_correct / len(data_loader.dataset)

def fixed_point_quantize(x, wl, fl, clamp=True, symmetric=True):
    scale = 2**(-fl)
    if symmetric:
        min_val = -2**(wl - fl - 1)
        max_val = 2**(wl - fl - 1) - scale
    else:
        min_val = -2**(wl - fl - 1) + scale
        max_val = 2**(wl - fl - 1) - scale
    
    if clamp:
        x = torch.clamp(x, min_val, max_val)
    
    x_scaled = x / scale
    x_rounded = torch.round(x_scaled).to(torch.int8)
    return x_rounded

def original_float32_model(model):
    file_name = f'float32_model.pth'
    torch.save(model.state_dict(), file_name)
    ## simple code to change pth to npy..
    torchdict = torch.load(file_name)
    numpydict = {}
    numpydict['fc1w'] = np.array(torchdict['fc1.weight'])
    # numpydict['fc1b'] = np.array(torchdict['fc1.bias'])

    numpydict['fc2w'] = np.array(torchdict['fc2.weight'])
    # numpydict['fc2b'] = np.array(torchdict['fc2.bias'])

    numpydict['fc3w'] = np.array(torchdict['fc3.weight'])
    # numpydict['fc3b'] = np.array(torchdict['fc3.bias'])

    file_name_2 = f'float32_model.npy'
    np.save(file_name_2, numpydict, allow_pickle=True)

def quantized_model_with_wl_87654321(model):
    best_accuracy = 0
    best_fl = 0
    for wl in range(1, 9): # 1 ~ 8
        for fl in range(1, wl+1):
            quantized_model = copy.deepcopy(model)
            for name, param in quantized_model.named_parameters():
                quantized_param = fixed_point_quantize(param.data, wl, fl=fl)
                param.data.copy_(quantized_param)
            
            accuracy = evaluate_model(quantized_model, test_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_fl = fl
                
        print(f"Best fl: {best_fl}, for wl = {wl} with best accuracy: {best_accuracy*100:.2f}%")

        quantized_model = copy.deepcopy(model)
        for name, param in quantized_model.named_parameters():
            quantized_param = fixed_point_quantize(param.data, wl, fl=best_fl)
            param.data.copy_(quantized_param)

        file_name = f'quantized_model_wl={wl}.pth'
        torch.save(quantized_model.state_dict(), file_name)
        ## simple code to change pth to npy..
        torchdict = torch.load(file_name)
        numpydict = {}
        numpydict['fc1w'] = np.array(torchdict['fc1.weight'])
        # numpydict['fc1b'] = np.array(torchdict['fc1.bias'])

        numpydict['fc2w'] = np.array(torchdict['fc2.weight'])
        # numpydict['fc2b'] = np.array(torchdict['fc2.bias'])

        numpydict['fc3w'] = np.array(torchdict['fc3.weight'])
        # numpydict['fc3b'] = np.array(torchdict['fc3.bias'])

        file_name_2 = f'quantized_model_wl={wl}.npy'
        np.save(file_name_2, numpydict, allow_pickle=True)
        
original_float32_model(model)
quantized_model_with_wl_87654321(model)