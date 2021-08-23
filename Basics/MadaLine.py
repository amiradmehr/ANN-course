import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from helper import *

''' initializing dataset for two clasess '''
X1 = [[0.5,2.5], [1,2], [1,3], [1.25,2.5], [1.5,2], [1.5,3]]
n_c1 = len(X1)
X2 = [[0,2], [0,3], [1,1], [1,4], [2,2], [2,4]]
n_c2 = len(X2)
X = X1 + X2
Y = [1]*n_c1 + [-1]*n_c2

datas = [0]*len(Y)
for i in range(len(Y)):
    datas[i] = X[i] + [Y[i]]

X1 = np.array(X1)
X2 = np.array(X2)
X = np.concatenate((X1, X2))

np.random.shuffle(datas)
datas = np.array(datas)
X = datas[:,0:2]
Y = datas[:,2]

''' creating a class for a single AdaLine neural network '''
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.010 * np.random.randn(n_inputs, n_neurons)
        # self.weights = np.ones((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
        # self.biases = 0.1 * np.random.randn(1, n_neurons)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return activation(self.output)


hidden = Layer(2,3)
out_neuron = Layer(3,1)
out_neuron.weights = np.array([[1],[1],[1]])
out_neuron.biases = np.array([[-2]])


training(X,Y,hidden,out_neuron,0.005,1000)

''' plotting datas '''
plt.plot(X1[:,0],X1[:,1], 'o', color = 'orange')
plt.plot(X2[:,0],X2[:,1], 'o', color = 'blue')
plt.grid()
linplot(hidden.weights, hidden.biases)
plt.show()



''' adding additional 94 points'''
X1_rand = np.random.normal(1, 0.2, 94)
X2_rand = np.random.normal(2.5, 0.2, 94)
Xrand = np.stack((X1_rand, X2_rand), axis=1)
X =  np.concatenate((X1,Xrand))
Y = np.concatenate((Y,np.array([[1]]*94)))


hidden = Layer(2,3)
out_neuron = Layer(3,1)
out_neuron.weights = np.array([[1],[1],[1]])
out_neuron.biases = np.array([[-2]])


training(X,Y,hidden,out_neuron,0.005,1000)

''' plotting datas '''
plt.plot(X1[:,0],X1[:,1], 'o', color = 'red')
plt.plot(X2[:,0],X2[:,1], 'o', color = 'blue')
plt.grid()
linplot(hidden.weights, hidden.biases)
plt.show()