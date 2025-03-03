import numpy as np
import time
from scipy.signal import convolve2d
from tqdm import trange

mnist = np.load("mnist-original.npy", allow_pickle=True)

X = mnist.item().get("data").T / 255
y = mnist.item().get("label")[0]

weights = np.load('model.npy', allow_pickle=True)

conv1w = weights.item().get('conv1w')
conv2w = weights.item().get('conv2w')
fc3w = weights.item().get('fc3w')

batch_size = 100

X1 = np.zeros((batch_size, 16, 26, 26))  # Temporary buffer for X1
X2 = np.zeros((batch_size, 16, 24, 24))  # Temporary buffer for X2

def avg_pool2d(x, kernel_size=2, stride=2):
    batch_size, channels, height, width = x.shape

    out_height = height // kernel_size
    out_width = width // kernel_size
    
    x_summed = np.add.reduceat(np.add.reduceat(x, np.arange(0, height, stride), axis=2), 
                               np.arange(0, width, stride), axis=3)
    
    x_pooled = x_summed / (kernel_size * kernel_size)
    
    return x_pooled

def feed_foward(X0):
    ## unfortunately, I found no efficient implementation of 2D Conv without using pytorch 
    ## this code is VERY SLOW. Just use this to see the correctness of the results 
    X0 = X0.reshape(-1, 1, 28, 28)

    ## conv1 layer
    for b in range(batch_size):
        for co in range(16):
            X1[b, co] = convolve2d(X0[b, 0], conv1w[co, 0], mode='valid')

    ## ReLU        
    X1[X1 < 0] = 0

    ## conv2 layer
    for b in range(batch_size):
        for co in range(16):
            for ci in range(16):
                X2[b, co] = convolve2d(X1[b, ci], conv2w[co, ci], mode='valid')    

    X2[X2 < 0] = 0
    A2 = avg_pool2d(X2)
    A2 = A2.reshape(-1, 2304)
    X3 = np.matmul(A2, fc3w.T)
    return X3

start_time = time.time()  # Start timing

prediction = []
for idx in trange(len(X) // batch_size):
    xs = X[batch_size * idx:batch_size * idx + batch_size]
    ys = y[batch_size * idx:batch_size * idx + batch_size]
    outputs = feed_foward(xs)
    for output, yk in zip(outputs, ys):
        prediction.append(np.argmax(output) == (yk))
    # if idx % 20 == 0:
    #     print(f"Batch {idx}, Current Accuracy: {np.mean(prediction) * 100:.2f}%")

end_time = time.time()  # End timing

score = np.mean(prediction) * 100

print(f"Final Accuracy: {score:.2f}%")
print(f"Total Inference Time: {end_time - start_time:.2f} seconds")
