import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from torchvision import datasets, transforms

# Load MNIST and model weights
mnist = np.load("mnist-original.npy", allow_pickle=True)
X = mnist.item().get("data").T / 255
y = mnist.item().get("label")[0]

weights = np.load("binary_model_uint8_-11.npy", allow_pickle=True)
conv1w = torch.tensor(weights.item().get("conv1w"), dtype=torch.float32)  # Shape: [16, 1, 3, 3]
conv2w = torch.tensor(weights.item().get("conv2w"), dtype=torch.float32)  # Shape: [16, 16, 3, 3]
fc3w = torch.tensor(weights.item().get("fc3w"), dtype=torch.float32)      # Shape: [10, 2304]
# weight는 이진화가 되어있음

#print(conv1w)
#print(conv2w)
#print(fc3w)


batch_size = 100

def binary_quantization(input_array, threshold=0.3, binary_range=(-1.0, 1.0)):
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


# Avg Pooling function
def avg_pool2d(x, kernel_size=2, stride=2):
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)

# Feed-forward function
def feed_forward(X0):
    X0 = binary_quantization(X0)
    X0 = torch.tensor(X0.reshape(-1, 1, 28, 28), dtype=torch.float32)  # Shape: [batch_size, 1, 28, 28]

    # Conv1
    X1 = F.conv2d(X0, conv1w)  # Shape: [batch_size, 16, 26, 26]
    X1 = binary_quantization(X1, threshold=0.0)

    # Conv2
    X2 = F.conv2d(X1, conv2w)  # Shape: [batch_size, 16, 24, 24]
    X2 = binary_quantization(X2, threshold=0.0)

    # Avg Pooling
    A2 = avg_pool2d(X2, kernel_size=2, stride=2)  # Shape: [batch_size, 16, 12, 12]
    A2 = binary_quantization(A2, threshold=0.0)
    A2 = A2.view(A2.size(0), -1)  # Flatten to [batch_size, 2304]

    # Fully connected layer
    X3 = A2 @ fc3w.T  # Shape: [batch_size, 10]
    return X3

# Prediction loop
prediction = []
for idx in trange(len(X)//batch_size):
    xs = X[batch_size * idx: batch_size * idx + batch_size]
    ys = y[batch_size * idx: batch_size * idx + batch_size]

    outputs = feed_forward(xs)
    predicted_labels = outputs.argmax(dim=1).numpy()
    prediction.extend(predicted_labels == ys)

    if idx % 20 == 0:
        print(f"Batch {idx}: Accuracy = {np.mean(prediction) * 100:.2f}%")

score = np.mean(prediction) * 100
print(f"Final Accuracy: {score:.2f}%")