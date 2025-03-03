import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def binary_quantization(input_array, threshold=0.2, binary_range=(-1.0, 1.0)):
    binary_min, binary_max = binary_range
    
    # PyTorch 텐서인지 확인하고 NumPy로 변환
    if isinstance(input_array, torch.Tensor):
        input_array = input_array.numpy()
    
    # threshold가 텐서일 경우 스칼라 값으로 변환
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.item()
    
    # 이진 양자화 수행
    quantized_array = np.where(input_array >= threshold, binary_max, binary_min)
    
    # PyTorch 텐서로 반환
    return torch.tensor(quantized_array, dtype=torch.float32)

# Binary Activation (Applied STE for sign function)
class BinaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.sign(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1 - torch.tanh(input) ** 2)
        #grad_input = grad_output * (torch.abs(input) <= 1).float()
        #grad_input = grad_output * (1 - input / (1 + torch.abs(input)) ** 2)
        return grad_input

class BinaryWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        return torch.sign(weights)  # Forward -> binarize weights

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Backward -> apply STE


class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(16)
        #self.dropout = nn.Dropout(0.5)
        #self.dropout = nn.Dropout(0.5)

        # Fully connected 레이어 크기 계산
        dummy_input = torch.zeros(1, 1, 28, 28)
        out = self._forward_features(dummy_input)
        #print(out.view(-1).size(0))
        self.fc = nn.Linear(out.view(-1).size(0), 10, bias=False)

    def _forward_features(self, x):
        x = binary_quantization(x)
        
        binary_conv1_weight = BinaryWeights.apply(self.conv1.weight)
        x = F.conv2d(x, binary_conv1_weight, stride=1)
        #x = F.relu(x)
        x = BinaryActivation.apply(x)

        binary_conv2_weight = BinaryWeights.apply(self.conv2.weight)
        x = F.conv2d(x, binary_conv2_weight, stride=1)
        #x = F.relu(x)
        x = BinaryActivation.apply(x)

        x = F.avg_pool2d(x, 2)
        x = BinaryActivation.apply(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)

        # Fully connected 레이어 가중치 이진화
        binary_fc_weight = BinaryWeights.apply(self.fc.weight)
        x = F.linear(x, binary_fc_weight)
        return F.log_softmax(x, dim=1)

# Weight Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

# Dataset load / Transform definition
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])
"""
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])"""

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model, loss function, and optimizer
model = BinaryNN()
initialize_weights(model)
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


torch.save(model.state_dict(),'model.pth')

## simple code to change pth to npy..
torchdict = torch.load('model.pth')
numpydict = {}
numpydict['conv1w'] = np.array(torchdict['conv1.weight'])

numpydict['conv2w'] = np.array(torchdict['conv2.weight'])

numpydict['fc3w'] = np.array(torchdict['fc.weight'])

np.save('model.npy', numpydict, allow_pickle=True)


