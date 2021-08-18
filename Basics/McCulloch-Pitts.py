import numpy as np


def myAND(x):
    weights = np.array([[1, 1]]).T
    y = np.dot(x, weights)
    threshold = 2
    out = np.heaviside(y-threshold, 1)
    return out

def myXOR(x):
    weights1 = np.array([[2,-1],[-1,2]]).T
    weights2 = np.array([[2, 2]]).T
    threshold = 2
    layer1 = np.dot(x, weights1)
    layer1_active = np.heaviside(layer1-threshold, 1)
    layer2 = np.dot(layer1_active, weights2)
    out = np.heaviside(layer2-threshold, 1)
    return out

def half_adder(x):
    s = myXOR(x)
    c = myAND(x)
    return c, s

def mymultiplier(A,B):
    P = np.empty([4,])
    c = 0
    AA = A[::-1]
    BB = B[::-1]
    P[0] = myAND([AA[0],BB[0]])
    c, P[1]= half_adder([int(myAND([AA[0],BB[1]])),
                         int(myAND([AA[1],BB[0]]))])
    P[3], P[2]= half_adder([int(myAND([AA[1],BB[1]])),
                            int(c)])
    return P[::-1]

bits = [[0, 0],
     [0, 1],
     [1,0],
     [1,1]]

for i in bits:
    for j in bits:
        a =mymultiplier(i,j)
        print(i)
        print(j)
        print('-------')
        print(a)
        print('\n')

