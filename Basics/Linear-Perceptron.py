import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = [[-1,-1],
     [-1,1],
     [1,-1],
     [1,1]]
Y = [1,
     1,
     1,
     -1]
class Perceptron:
    def __init__(self, l_rate = 0.01, n_epochs = 10, n_inputs = 2):
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.n_inputs = n_inputs

    def activation(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.output>0:
            return 1
        elif self.output ==0:
            return 0
        return -1

    def training(self, training_inputs, classes):
        self.weights = np.zeros((self.n_inputs, 1))
        self. biases = np.random.rand(1,1)
        self.misclassified_epoch = []

        for epochs in range(self.n_epochs):
            print('-----epoch %d -----' %(epochs+1))
            data = {
                'inputs':[],
                'expected output':[],
                'current weight':[],
                'current bias':[],
                'net':[],
                'activation output':[],
                'updated weight':[],
                'updated bias':[]}
            misclassified = 0

            for inputs, label in zip(training_inputs, classes):
                active = self.activation(inputs)
                data['activation output'].append(active)
                net = np.dot(inputs, self.weights) + self.biases
                data['net'].append(net)
                data['current weight'].append(self.weights)
                data['current bias'].append(self.biases)
                err = label - active

                if err != 0:
                    misclassified += 1
                    #updating weights and biases
                    inputs_ = np.array([inputs]).T
                    alpha_t = self.l_rate * label
                    self.weights = np.add(self.weights,inputs_ * alpha_t)
                    self.biases = np.add(self.biases, alpha_t)
                data['inputs'].append(inputs)
                data['expected output'].append(label)
                data['updated weight'].append(self.weights)
                data['updated bias'].append(self.biases)

            self.misclassified_epoch.append(misclassified)
            print(pd.DataFrame(data))
            if(misclassified==0):
                pass

        return self.weights, self.biases


myneuron = Perceptron(n_epochs=5,l_rate=1)
weight, bias = myneuron.training(X,Y)

#plotting the line and points
x = np.linspace(-1,1)
y = -x * weight[0]/weight[1] - bias[0]/weight[1] 
plt.plot(x,y)
x = []
y = []
for _ in X:
    x.append(_[0])
    y.append(_[1])
plt.plot(x,y, 'o', color='black')
plt.show()

#plotting misclassification over each epoch
x = list(range(1,len(myneuron.misclassified_epoch)+1))

plt.plot(x,myneuron.misclassified_epoch)
plt.xlabel('epoch')
plt.ylabel('misclassification')
plt.show()
print(weight.shape)