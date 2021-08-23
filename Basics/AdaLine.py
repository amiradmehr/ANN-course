import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


''' initializing dataset for three clasess '''
std, n1, n2, n3 = 0.2, 100, 100, 200
X1_1 = np.random.normal(1, std, n1)
X2_1 = np.random.normal(1, std, n1)
X1 = np.stack((X1_1, X2_1), axis=1)

X1_2 = np.random.normal(2.5, std, n2)
X2_2 = np.random.normal(2, std, n2)
X2 = np.stack((X1_2, X2_2), axis=1)

X1_3 = np.random.normal(1, std, n3)
X2_3 = np.random.normal(3, std, n3)
X3 = np.stack((X1_3, X2_3), axis=1)

''' concatenating all classes into one matrix '''
X = np.concatenate((X1, X2, X3))

''' initializing each desired output for each set '''
Y_1 = [1]*n1 + [-1]*n2 + [-1]*n3
Y_2 = [-1]*n1 + [1]*n2 + [-1]*n3
Y_3 = [-1]*n1 + [-1]*n2 + [1]*n3

''' creating a class for a single AdaLine neural network '''
class AdaLine:

    ''' initial method '''
    def __init__(self, l_rate = 0.0006, n_epochs = 500, n_inputs = 2):
        self.n_epochs = n_epochs
        self.l_rate = l_rate
        self.n_inputs = n_inputs

    ''' activation function '''
    def activation(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if self.output>=0:
            return 1
        return -1

    ''' training dataset '''
    def training(self, training_inputs, classes, threshold=1):

        ''' initializing weights and biases '''
        self.weights = np.zeros((self.n_inputs, 1))
        self. biases = np.random.rand(1,1)
        self.cost = []
        col_classes = np.array(classes).reshape(n1+n2+n3,1)
        ''' loop through each epoch '''
        for epochs in range(self.n_epochs):
            cost_ = 0

            ''' loop through each epochs datas '''
            for inputs, label in zip(training_inputs, classes):
                net = np.dot(inputs, self.weights) + self.biases
                err = float(label - net)
                
                # cost_ += (err**2)
                # error.append((err**2))
                ''' updating weights and biases '''
                inputs_ = np.array([inputs]).T
                alpha_t = self.l_rate * (err)
                self.weights = np.add(self.weights,inputs_ * alpha_t)
                self.biases = np.add(self.biases, alpha_t)
    
            whole_net = np.dot(training_inputs, self.weights) + self.biases
            all_errors = np.square(np.subtract(col_classes, whole_net))
            cost_ = np.sum(all_errors)
            self.cost.append(cost_)
            if all(i <= threshold for i in all_errors):
                print("epochs --> %d" %epochs)
                break
        return self.weights, self.biases    
            # '''
            # This part is to see each update cycle in a plot 
            # '''
            # x = np.linspace(0,3)
            # y = -x * self.weights[1]/self.weights[0] - self.biases[0]/self.weights[0]
            # plt.plot(x,y)
            # plt.show(block=False)
            # plt.pause(0.001)
            # plt.clf()
            # plt.plot(X1_1,X2_1, 'o', color='red')
            # plt.plot(X1_2,X2_2, 'o', color='blue')
            # plt.plot(X1_3,X2_3, 'o', color='green')

        
    

adaline_network_1 = AdaLine()
weight1, bias1 = adaline_network_1.training(X,Y_1,threshold=0.7)

adaline_network_2 = AdaLine()
weight2, bias2 = adaline_network_2.training(X,Y_2,threshold=0.7)

adaline_network_3 = AdaLine()
weight3, bias3 = adaline_network_3.training(X,Y_3,threshold=0.7)


''' plotting three classes of data '''
plt.plot(X1_1,X2_1, 'o', color='red')
plt.plot(X1_2,X2_2, 'o', color='orange')
plt.plot(X1_3,X2_3, 'o', color='green')
# plt.legend(['class #1', 'class #2', 'class #3'])
''''''''''''''''''''''''''''''''''''

''' plotting line for classes '''
x = np.linspace(X.min(),X.max())
m1 = weight1[0]/weight1[1]
b1 = bias1[0]/weight1[1]

m2 = weight2[0]/weight2[1]
b2 = bias2[0]/weight2[1]

m3 = weight3[0]/weight3[1]
b3 = bias3[0]/weight3[1]

y1 = -x * m1 - b1
y2 = -x * m2 - b2
y3 = -x * m3 - b3

# plt.figure(figsize=(3,4))
plt.ylim(X.min()*0.8, X.max()*1.1)
plt.xlim(X.min()*0.8,X.max()*1.1)
plt.plot(x,y1, color='red')
plt.plot(x,y2, color='orange')
plt.plot(x,y3, color='green')
plt.legend(['class #1', 'class #2', 'class #3',
            'class #1 line', 'class #2 line', 'class #3 line'])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
''''''''''''''''''''''''''''''''''''

''' plotting cost in each epoch '''
x1 = list(range(1,len(adaline_network_1.cost)+1))
x2 = list(range(1,len(adaline_network_2.cost)+1))
x3 = list(range(1,len(adaline_network_3.cost)+1))
plt.plot(x1,adaline_network_1.cost)
plt.plot(x2,adaline_network_2.cost)
plt.plot(x3,adaline_network_3.cost)
plt.legend(['Cost 1', 'Cost 2', 'Cost 3'])
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()
''''''''''''''''''''''''''''''''''''