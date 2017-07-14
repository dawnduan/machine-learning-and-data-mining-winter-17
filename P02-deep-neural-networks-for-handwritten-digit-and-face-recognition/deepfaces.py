################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

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
        M[key] = partXread(M[key])
    return M

def partXread(sset): 
    '''return dim(x) = m * n
    '''
    x = ones(0)
    for a in sset:
        # read each img in trainingSet into 2d array
        a = './part10/'+a
        im1 = (imread(a)[:,:,:3]).astype(float32)
        im1 = im1 - mean(im1)
        im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]        
        # append x1 up to x1024
        if (len(x) == 0): 
            x = temp 
        else:
            # vstack them
            x = vstack((x, temp))
    return x


################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random


import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]



################################################################################
#Read Image, and change to BGR


im1 = (imread("laska.png")[:,:,:3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]


################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



x = tf.placeholder(tf.float32, (None,) + xdim)


#conv1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
conv1W = tf.Variable(net_data["conv1"][0])
conv1b = tf.Variable(net_data["conv1"][1])
conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
conv1 = tf.nn.relu(conv1_in)

#lrn1
#lrn(2, 2e-05, 0.75, name='norm1')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool1
#max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
conv2W = tf.Variable(net_data["conv2"][0])
conv2b = tf.Variable(net_data["conv2"][1])
conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv2 = tf.nn.relu(conv2_in)


#lrn2
#lrn(2, 2e-05, 0.75, name='norm2')
radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)

#maxpool2
#max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
conv3W = tf.Variable(net_data["conv3"][0])
conv3b = tf.Variable(net_data["conv3"][1])
conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv3 = tf.nn.relu(conv3_in)

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
conv4W = tf.Variable(net_data["conv4"][0])
conv4b = tf.Variable(net_data["conv4"][1])
conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
conv4 = tf.nn.relu(conv4_in)#<tf.Tensor 'Relu_3:0' shape=(?, 13, 13, 384) dtype=float32>

num_iter = 1220
num_img = 420
nhid = 650

#if tf.Tensor.get_shape(conv4)[0] == Dimension(None):
    #print('lll')
    #x_conv = tf.reshape(conv4,[13*13*384,])
#else:
    #x_conv = tf.reshape(conv4,[13*13*384,int(tf.Tensor.get_shape(conv4)[0])])    

#nhid = 650

##Overfitting, nhid = 650, > 1220 num_iter start to overfitting
#W0 = tf.Variable(tf.random_normal([nhid, 13*13*384], stddev=0.1))
#b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))

#W1 = tf.Variable(tf.random_normal([nhid, 2], stddev=0.01))

#b1 = tf.Variable(tf.random_normal([2], stddev=0.01))

#layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
#layer2 = tf.matmul(layer1, W1)+b1


#y = tf.nn.softmax(layer2)
#y_ = tf.placeholder(tf.float32, [None, 6]) 
##x<tf.Tensor 'Placeholder:0' shape=(?, 1024) dtype=float32>
#W0 = tf.Variable(tf.random_normal([13, 13, 384, nhid], stddev=0.1)) #sess.run(W0).shape(1024, 650)
#b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))#(650,)

#W1 = tf.Variable(tf.random_normal([13,13,nhid, 2], stddev=0.01))#(650, 6)

#b1 = tf.Variable(tf.random_normal([2], stddev=0.01))#(6,)

#layer1 = tf.nn.tanh(tf.matmul(conv4, W0)+b0)#<tf.Tensor 'Tanh:0' shape=(?, 650) dtype=float32>
##<tf.Tensor 'Tanh_1:0' shape=(13, 13, 13, 650) dtype=float32>
#layer2 = tf.matmul(layer1, W1)+b1#<tf.Tensor 'add_1:0' shape=(?, 6) dtype=float32>


##y = tf.nn.softmax(layer2)#<tf.Tensor 'Softmax:0' shape=(?, 6) dtype=float32>
##y_ = tf.placeholder(tf.float32, [None, 2])#<tf.Tensor 'Placeholder_1:0' shape=(?, 6) dtype=float32>

#prob = tf.nn.softmax(layer2)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

t = time.time()
#output = sess.run(prob, feed_dict = {x:[im1,im2]})#[2,13,13,384] vs [13,13,384,650]
################################################################################

#Output:
#for input_im_ind in range(output.shape[0]):
    #inds = argsort(output)[input_im_ind,:]
    #print("Image", input_im_ind)
    #for i in range(5):
        #print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])

#print(time.time()-t)

