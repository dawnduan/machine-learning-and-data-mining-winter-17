from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import scipy.io

import cPickle

import os
from scipy.io import loadmat

import math

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
    
np.random.seed(411)

# Mike's code.
def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def cross_entropy(y, y_):
    return -sum(y_*log(y))

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )
    
    
##Part1:Selects 10 random images from each training set.
def part1(data):
    path = os.getcwd()
    imgs = path + "/number_images/"
    
    entries = os.listdir(path)
    
    if ("number_images" not in entries):
        os.mkdir(imgs)
        
    for number in data.keys():
        if ("__" in number):
            continue
            
        if ("test" in number):
            continue
            
        for i in range(10):
            rand = random.randint(0, len(M[number]))
            img = M[number][rand].reshape((28, 28))
            filename = "{}_rand{}.jpg".format(number, i)
            cmap = cm.gray
            img = cmap(img)
            imsave(imgs+filename, img)
            
##PART2:softmax
def net_w(x, w):
    b = ones((1000, 10)) 
    return (dot(x, w) + b).T 
    
##PART3 
def cost(x, y, w): 
    return (-1)*sum(log(softmax(net_w(x, w))))
    
def dcost(x, y, w):
    return dot((softmax(net_w(x, w)) - y.T),x).T
    
    
def finiteDiff(x, y, theta):
    EPS = 1e-10
    grad = zeros((theta.shape)) #construct a n*k matrix
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            J0 = cost(x, y, theta) # J(p,y)
            theta[i][j] += EPS
            diff = cost(x, y, theta) - J0
            grad[i][j] = float(diff)/EPS
    return grad
    

##PART4    
def grad_descent(f, df, x, y, init_t, alpha, testx, testy):
    EPS = 1e-10
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 5000
    iter  = 0
    while iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
           # print f(x, y, t)
           # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (x[0,0], x[0,1], x[0,2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t), "\n "
            #add code here
            print("testingdata")
            print (1- performance(x,y,t,1000))
            trainingset.append(1- performance(x,y,t,1000))
            print("trainingdata")
            print (1- (performance(testx, testy, t,1000)))
            testingset.append(1- (performance(testx, testy, t,1000)))
            test = df(x, y, t)
        iter += 1
    return t


def performance(x, y , w, size_of_data):

    hitnum = 0.0    
    for i in range(size_of_data):
        guess = np.where(softmax(net_w(x,w)).T[i] == amax(softmax(net_w(x,w)).T[i]))[0][0]

        result = np.where(y[i] == amax(y[i]))[0][0]
        
        if guess == result:
            hitnum += 1.0

    return hitnum / float(size_of_data)
    
    
def batch(datasize):# 5 - 50
    new_x = [] #sizeoft * 784
    new_y = [] #10 * sizeoft
    for i in range(10):
        x_temp = M["train"+str(i)][:datasize*random.randint(0,50),:]/255.0 
        y_temp = zeros((datasize,10))
        y_temp[:,i] = 1
        if i == 0:
            x = x_temp
            y = y_temp
        else:
            x = vstack((x,x_temp))
            y = vstack((y,y_temp))          
    return new_x,new_y
    
##PART5

def f(x, y, theta):
    return sum( (y - dot(x,theta)) ** 2)

def df(x, y, theta):
    return -2*dot( x.T,(y-dot(x, theta)))
    

    ##Main
if __name__ == "__main__":
    M = loadmat("mnist_all.mat")
    #part1(M)
    ##part2
    '''
    output = []
    w = 0.0 * np.ones((28*28,))
    w[:] = np.random.rand()
    y = [i for i in range(10)] 
    for i in range(10):
        x_temp = M["train"+str(i)][:100,:]/255.0
        o_i = net_w(x_temp, w)
        output.append(o_i) # 100 img per digit # m = 900
    output = np.array(output) # [10x100]
    prob = softmax(output)
    '''
    ##part3 
    
    x = [] #sizeoft * 784
    y = [] #10 * sizeoft
    w = [] #shape (784,10) .
    w_temp = np.ones((28*28,)) 
    w_temp[:] = np.random.rand()
    for i in range(10): # every label
        x_temp = M["train"+str(i)][:100,:]/255.0
        y_temp = zeros((100,10))
        y_temp[:,i] = 1
        if i == 0:
            x = x_temp
            y = y_temp
            w = w_temp

        else:
            x = vstack((x,x_temp))
            y = vstack((y,y_temp))
            w = vstack((w,w_temp))
            
    w = w.T
    alpha = 1e-5
   
    
   # size 250
    #weight = grad_descent(cost, dcost, x, y, w, alpha)
    #print performance(x,y,weight, 10000) #train
    
    
    ##Part4 Test test data
    
    
    testx = []
    testy = []
    for i in range(10): # every label
        x_temp = M["test"+str(i)][:100,:]/255.0 
        y_temp = zeros((100,10))
        y_temp[:,i] = 1
        if i == 0:
            testx = x_temp
            testy = y_temp

        else:
            testx = vstack((testx,x_temp))
            testy = vstack((testy,y_temp))
    
    
    trainingset = []
    testingset = []
    
    
    #grad_descent(cost, dcost, x, y, w, alpha, testx, testy)
    
    

    # num_iter = 5000
    # x = np.arange(0, num_iter, 500) # num_iter
   
#   #   
    # train = plt.plot(x,trainingset,'g--')
    # test = plt.plot(x,testingset,'r--')
    # 

    iterslist = []
    for i in range(0,5000,500):
        iterslist.append(i)
    print trainingset
    print testingset

    #To display the graph of learning curve
    trainingset = [0.366, 0.14400000000000002, 0.11599999999999999, 0.10299999999999998, 0.09099999999999997, 0.07699999999999996, 0.07399999999999995, 0.06899999999999995, 0.06499999999999995, 0.05900000000000005]
    
    testingset = [0.40900000000000003, 0.22799999999999998, 0.19999999999999996, 0.18300000000000005, 0.18100000000000005, 0.17500000000000004, 0.16900000000000004, 0.16600000000000004, 0.16000000000000003, 0.15800000000000003]

    train = plt.plot(iterslist,trainingset)    
    test = plt.plot(iterslist, testingset)
    plt.axis([0, 5000, 0, 1])
    plt.legend([test, train], ["test", "train"])
   
    plt.show()
    
   # print performance(testx, testy, weight, 2000) #test
    
 
    # plt.show()


    ##Part5

    
    #Part 4 has testing data
    # noise_x = [] #sizeoft * 784
    # noise_y = [] #10 * sizeoft
    # noise_w = [] #shape (784,10) .
    # w_temp = np.ones((28*28,)) 
    # w_temp[:] = np.random.rand()
    # for i in range(10): 
    #     x_temp = M["train"+str(i)][:90,:]/255.0 
    #     y_temp = zeros((90,10))
    #     
    #     y_temp[:,i] = 1
    #     if i == 0:
    #         noise_x = x_temp
    #         noise_y = y_temp
    #         noise_w = w_temp
    #     
    #     else:
    #         noise_x = vstack((noise_x,x_temp))
    #         noise_y = vstack((noise_y,y_temp))
    #         noise_w = vstack((noise_w,w_temp))
    # 
    # 
    #     x_temp = M["train"+str(i)][90:100,:]/255.0 
    #     y_temp = zeros((10,10))
    #     y_temp[:,i-5] = 1

   ##       noise_x = vstack((noise_x,x_temp))
    #     noise_y = vstack((noise_y,y_temp))

   ##   
    # noise_w = noise_w.T
    # 
    # w_logistireg = grad_descent(cost, dcost, noise_x, noise_y, noise_w, alpha) 
    # w_lineareg = grad_descent(f, df, noise_x, noise_y, noise_w, alpha)
    # print("Using bad tranining data set to training")
    # print("logisti Regression")
    # print performance(noise_x,noise_y,w_logistireg, 1000) 
    # print("linear Regression")
    # print performance(noise_x,noise_y,w_lineareg, 1000)
    # 
    # print("Testing data")
    # print("logistic Regression ")
    # print performance(testx,testy,w_logistireg, 1000)  
    # print("linear Regression")
    # print performance(testx,testy,w_lineareg, 1000) 
    # 

   ##   
    # ##PART6 On Paper 


    
