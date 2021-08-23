import numpy as np
import matplotlib.pyplot as plt



def linplot(weight, bias):
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    xx = x.reshape(2,1)
    m = -weight[0]/weight[1]
    col = len(weight[0])
    mm = m.reshape(1, col)
    y = xx * mm - bias/weight[1]
    for i in range(col):
        plt.plot(x, y[:,i], color = 'green')


def activation(out_in):
    return np.where(out_in >= 0, 1, -1)

def training(inputs, target, hidden_layer, out_layer, l_rate, n_epoch):

    for i in range(n_epoch):
        for s,t in zip(inputs, target):
            z = hidden_layer.forward(s)
            y = out_layer.forward(z)
            z_in = hidden_layer.output
            
            if y != t:
                if t == -1:
                    index = find_pos(z_in)
                    for jj in index:
                        hidden_layer.weights[:,jj] += l_rate * (-1 - z_in[0][jj]) * s
                        hidden_layer.biases[:,jj] += l_rate * (-1 - z_in[0][jj])

                elif t == 1:
                    elmnt = min((abs(j), j) for j in z_in[0])[1]
                    if elmnt < 0:
                        indx = list(z_in[0]).index(elmnt)
                        hidden_layer.weights[:,indx] += l_rate * (1 - z_in[0][indx]) * s
                        hidden_layer.biases[:,indx] += l_rate * (1 - z_in[0][indx])
        
def find_pos(lst):
    l = list(list(lst[0]))
    index = []
    for i in range(len(l)):
        if l[i]>0:
            index.append(i)
    return index

# def find_pos(lst):
#     l = list(list(lst[0]))
#     for i in range(len(l)):
#         if l[i]>0:
#             return i
# def act_func(inputs, weights, biases):
#     output = np.dot(inputs, weights) + biases
#     return np.where(output >= 0, 1, -1)

# def line_plot(weight1, weight2, bias, colour):
#     axes = plt.gca()
#     x = np.array(axes.get_xlim())
#     y = -(weight1/weight2) * x - (bias/weight2)
#     for i in range(len(y)):
#         plt.plot(x,y[i],color = colour)