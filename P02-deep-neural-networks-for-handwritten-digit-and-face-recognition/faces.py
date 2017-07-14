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

import pickle as cPickle

import os
from scipy.io import loadmat
import tensorflow as tf


act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']    
testSetSz = 30
trSetSz = 90
validSetSz = 30



def get_train_batch(M, N, act):
    n = N/6
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, len(act)))
    
    train_k =  ['train_'+a.split()[1].lower() for a in act] # key

    train_size = len(M[train_k[0]])
    #train_size = 5000
    
    for k in range(len(act)):
        train_size = len(M[train_k[k]])
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(len(act))
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s
    
def get_train(M, act):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, len(act)))
    
    train_k = ['train_'+a.split()[1].lower() for a in act] # key
    for k in range(len(act)):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(len(act))
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_test(M, act):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, len(act)))
    
    test_k = ['test_'+a.split()[1].lower() for a in act] # key
    for k in range(len(act)):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(len(act))
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_valid(M, act):
    batch_xs = zeros((0, 32*32))
    batch_y_s = zeros( (0, len(act)))
    
    valid_k = ['valid_'+a.split()[1].lower() for a in act] # key
    for k in range(len(act)):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(len(act))
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s
   
def formM(act, trSetSz, validSetSz, testSetSz):
    M = dict()
    dataset = []
    for a in act:
        name = a.split()[1].lower()
        #print(os.listdir('.'))
        lis = [item for item in os.listdir('./uncropped') if name in item] # needed to be where images lie
        #print(lis)
        #dataset.append(lis) #output a lis where contains all the images has 'name' - last name of the actor
        random.shuffle(lis) #randomized dataset 
        # Dividing [:100] Training, []
        l = len(lis)
        M['train_'+name] = lis[:trSetSz]
        M['test_'+name] = lis[(l-testSetSz):]
        M['valid_'+name] = lis[(l-2*validSetSz):(l-testSetSz)]
        
    for key in M.keys():
        M[key] = partVread(M[key])
    return M
 
def partVread(set): 
    '''return dim(x) = m * n
    '''
    x = ones(0)
    for a in set:
        # read each img in trainingSet into 2d array
        a = './uncropped/'+a
        temp = imread(a, True, 'L')
        temp = temp.flatten() 
        # Normalized for the images
        temp = temp - 127
        temp = temp /255.0  
        # append x1 up to x1024
        if (len(x) == 0): 
            x = temp 
        else:
            # vstack them
            x = vstack((x, temp))
    return x

def addPoint(t, v, tr, t_val, v_val, tr_val):
    t.append(t_val)
    v.append(v_val)
    tr.append(tr_val)
    return t, v, tr
#######################
M = formM(act, trSetSz, validSetSz, testSetSz)   
x = tf.placeholder(tf.float32, [None, 1024])
num_iter = 12#20
num_img = 420
nhid = 650

#Overfitting, nhid = 650, > 1220 num_iter start to overfitting

W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.1))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))

b1 = tf.Variable(tf.random_normal([6], stddev=0.01))

layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1


y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6]) #Inserts a placeholder for a tensor that will be always fed. wait for feed_dict()

#You can train the network by defining a cost function (itself a tensor), and defining a training set (an analog of an iteration of gradient descent). In the handout, I am using Adam, a more efficient variant of gradient descent which we'll discuss this week.

lam = 0.003
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))

reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty

train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)

#To run the optimization, I need to start a session of TensorFlow:
init = tf.initialize_all_variables()#to tf.global_variables_initializer() -Returns an Op that initializes global variables 
sess = tf.Session()# Launch the graph in a session.
sess.run(init) # Evaluate the tensor `init`

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_x, test_y = get_test(M, act)
valid_x, valid_y = get_valid(M, act)

# run the training step a bunch of times, plugging in values into placeholders


er_test = []
er_train = []
er_valid = []

for i in range(num_iter):
    #print i  
    batch_xs, batch_ys = get_train_batch(M, num_img, act)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    
    
    if i % 100 == 0:
        print("i=",i)
        t_val = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        print("Test:", t_val)
        #batch_xs, batch_ys = get_train(M, act)
        v_val = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
        print("Valid:", v_val)# what is it?
        
        tr_val = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
        print("Train:", tr_val)
        print("Penalty:", sess.run(decay_penalty))
        er_test, er_valid, er_train = addPoint(er_test, er_valid, er_train, (1-t_val), (1-v_val), (1-tr_val))
        
        #snapshot = {}
        #snapshot["W0"] = sess.run(W0)
        #snapshot["W1"] = sess.run(W1)
        #snapshot["b0"] = sess.run(b0)
        #snapshot["b1"] = sess.run(b1)
        #cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "wb"))   
'''
Initially hidden layer is 60 and 180 iterations - performance stops at 50%
400 hidden layer seems to work consistently

when 70 img per actor 
Overfitting, nhid = 650, > 1220 num_iter start to overfitting
increase nhid and num_iter really helps to imporve performance but heavy computation cost

test 78% validation 85% training 98%

80 img per actor - nhid 960 - 2k num_iter: test result stays on 76%

######################### Part 8 ##############################################
once overfitting occured - where regularization is necessary
small training example, small size of parameters, more num_iter
20img per actor, 100 nhid, 500 num_iter # not used

now: 40 img per actor (240), 60 nhid, 15000 num_iter

start with lamda = 0.003
comment: performance on valid is substential if N = total testsize
valid error rate reduce from 22% to 16% about 8% improvement
Notice swap valid and testset - regenerate results

######################### Part 9 ##############################################
'''

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

#x = np.arange(0., num_iter) # num_iter
#x = [i for i in range(num_iter)]
#test = plt.plot(x, er_test, 'g--')
#valid = plt.plot(x, er_valid, 'r--')
#train = plt.plot(x, er_train, 'b--')
#plt.axis([0, num_iter, 0, 1])
#plt.legend([test, valid, train], ["test", "valid", "train"])
#plt.show()

################################################################################
##Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()   

##get the index at which the output is the largest
#ind_a0 = argmax(sess.run(W1)[:,0])#Fran Drescher
#ind_a4 = argmax(sess.run(W1)[:,4])#Bill Hader 

##heatmap = ax.imshow(sess.run(W0)[:,ind_a0].reshape((32,32)), cmap = cm.coolwarm)
#heatmap = ax.imshow(sess.run(W0)[:,ind_a4].reshape((32,32)), cmap = cm.coolwarm)
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
