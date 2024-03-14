import torch as tch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot
samples = 5000
#Let's divide the toy dataset into training (80%) and rest for validation.
train_split = int(samples*0.8)
#Create a dummy classification dataset
X, y = make_blobs(n_samples=samples, centers=2, n_features=64,
cluster_std=10, random_state=2020)
y = y.reshape(-1,1)
#Convert the numpy datasets to Torch Tensors
X,y = tch.from_numpy(X),tch.from_numpy(y)
X,y =X.float(),y.float()
#Split the datasets inot train and test(validation)
X_train, x_test = X[:train_split], X[train_split:]
Y_train, y_test = y[:train_split], y[train_split:]
#Print shapes of each dataset
print("X_train.shape:",X_train.shape)
print("x_test.shape:",x_test.shape)
print("Y_train.shape:",Y_train.shape)
print("y_test.shape:",y_test.shape)
print("X.dtype",X.dtype)
print("y.dtype",y.dtype)
Output[]
X_train.shape: torch.Size([4000, 64])
x_test.shape: torch.Size([1000, 64])

Y_train.shape: torch.Size([4000, 1])
y_test.shape: torch.Size([1000, 1])
X.dtype torch.float32
y.dtype torch.float32

#Define a neural network with 3 hidden layers and 1 output layer
#Hidden Layers will have 64,256 and 1024 neurons
#Output layers will have 1 neuron
class NeuralNetwork(nn.Module):
def __init__(self):
super().__init__()
tch.manual_seed(2020)
self.fc1 = nn.Linear(64, 256)
self.relu1 = nn.ReLU()
self.fc2 = nn.Linear(256, 1024)
self.relu2 = nn.ReLU()
self.out = nn.Linear(1024, 1)
self.final = nn.Sigmoid()
def forward(self, x):
op = self.fc1(x)
op = self.relu1(op)
op = self.fc2(op)
op = self.relu2(op)

op = self.out(op)
y = self.final(op)
return y

