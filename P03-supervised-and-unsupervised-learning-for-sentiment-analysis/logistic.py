from pylab import *
import time
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from scipy.io import *
import os
import re
import random
from scipy.misc import imread
from scipy.misc import imresize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import tensorflow as tf

###Part 1
M = dict()
trSetSz = 200
validSetSz = 200
testSetSz = 200

for rev in os.listdir('./review_polarity/txt_sentoken'):
    lis = os.listdir('./review_polarity/txt_sentoken/'+rev)
    random.shuffle(lis)
    l = len(lis)
    M['train_'+rev] = lis[:trSetSz]
    M['test_'+rev] = lis[(l-testSetSz):]
    M['valid_'+rev] = lis[(l-2*validSetSz):(l-testSetSz)]
    
for key in M.keys():
    temp = []
    for film in M[key]:
        # Remove punctuMation & lower case the words
        file = open('./review_polarity/txt_sentoken/'+key[-3:]+'/'+film, 'r')
        text = file.read().lower()
        # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
        text = re.sub('[^a-z\ \']+', " ", text)
        temp.append(list(text.split()))#list of words per set
        file.close()
    M[key] = temp #overwrite the words into dict.values()
    
#2 Bayes nets
tot_w = 2000

def wordlist(films):
    dlis = []
    for f in films:
        dlis += f
    return list(set(dlis))
    
## Part 4 - Logistic Regression Model
def generate_inputs(setType, k_vector):
    # Initialize k_vector for v movie reviews 
    x = zeros(len(k_vector)) 
    y_ = zeros(1)
    y_exp = 1 
    for key in [setType + '_pos', setType + '_neg']:
        for revWords in M[key]: 
            # Initialize a k vector for this review 
            row = zeros(len(k_vector))
            
            # Find and set index of word corresponding to k_vector index
            for word in revWords: 
                if (word in k_vector): 
                    row[k_vector.index(word)] = 1
            
            # Add review to x and generate corresponding y 
            x = vstack((x, row))
            y_ = vstack((y_, [y_exp]))
        y_exp = 0;
        
    # Remove dummy rows 
    return x[1:], y_[1:]
    
def obtain_k(): 
    # Find all words that appear in positive and negative training class 
    trPos = wordlist(M['train_pos']) 
    trNeg = wordlist(M['train_neg']) 
    k_vector = trPos + trNeg
    k_vector = list(set(k_vector))
    return k_vector

def compute_network(): 
    # Obtain data
    k_vector = obtain_k()
    train_x, train_y = generate_inputs("train", k_vector) 
    test_x, test_y = generate_inputs("test", k_vector)
    val_x, val_y = generate_inputs("valid", k_vector) 
    
    # Plotting data 
    points_test = []; 
    points_val = [];
    points_training = []; 
    
    # Generate network 
    x = tf.placeholder(tf.float32, [None, len(k_vector)])
    
    W0 = tf.Variable(tf.random_normal([len(k_vector), 1], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([1], stddev=0.01))
    
    layer1 = tf.matmul(x, W0)+b0
    
    
    y = tf.nn.sigmoid(layer1)
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    
    lam = 0.00000
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum((y_*tf.log(y)) + ((1-y_)*tf.log(1-y)))
    
    train_step = tf.train.GradientDescentOptimizer(0.000005).minimize(reg_NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print("Train:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))
    
    
    iter = 15
    for i in range(1500):
      #print i  
      
      #idx = np.random.permutation(len(test_x))[:30]
      sess.run(train_step, feed_dict={x: train_x, y_: train_y})
      #print(sess.run(W0, feed_dict={x: test_x, y_: test_y})[:10])
      
      if i % 100 == 0:
        print("i=", i)
        test_val = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
        print("Test:", test_val)
        points_test.append(1-test_val)
        
        val = sess.run(accuracy, feed_dict={x: val_x, y_: val_y})
        print("Validation:", val)
        points_val.append(1-val)
        
        train_val = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
        print("Train:", train_val)
        points_training.append(1-train_val)
        
    i = [x for x in range(iter)]
    plt.plot(i, points_test, 'r', i, points_val, 'b', i, points_training, 'g')
    plt.show()
    
