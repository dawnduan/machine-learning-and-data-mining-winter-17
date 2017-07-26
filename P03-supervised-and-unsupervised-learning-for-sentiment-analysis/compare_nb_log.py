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
trSetSz = 400
validSetSz = 400
testSetSz = 400

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
    y_ = zeros(2); 
    y_exp = np.append(ones(1), zeros(1)); 
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
            y_ = vstack((y_, y_exp))
        y_exp = np.append(zeros(1), ones(1));
        
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
    
    # Generate network 
    x = tf.placeholder(tf.float32, [None, len(k_vector)])
    
    W0 = tf.Variable(tf.random_normal([len(k_vector), 2], stddev=0.01))
    b0 = tf.Variable(tf.random_normal([2], stddev=0.01))
    
    layer1 = tf.matmul(x, W0)+b0
    
    
    y = tf.nn.softmax(layer1)
    y_ = tf.placeholder(tf.float32, [None, 2])
    
    
    lam = 0.005
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    train_step = tf.train.AdamOptimizer(0.00005).minimize(reg_NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    print("Train:", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    
    for i in range(5000):
      #print i  
      sess.run(train_step, feed_dict={x: train_x, y_: train_y})
      
      
      if i % 1000 == 0:
        print("i=", i)#Generate a dataset on which Logistic Regression works a significantly better than Naive Bayes. Explain how you accomplished this, and include the code you used to generate the dataset, as well as your experimental results, in your report. 
        print("Train:", sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
        print("Test:", sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))

    ## part 6
    findTop100L(sess.run(W0), k_vector) #for logistic regression


def findTop100L(theta, k_vector):
    ''' input a numpy array theta values, a copy of word corresponding to each theta value
    return the top 100 words list that contains highest theta value'''
    t_rank = theta[:]
    return [[k_vector[list(theta[:,j]).index(item)] for item in sorted(t_rank[:,j], reverse = True)[:100]] for j in range(theta.shape[1])]

def findTop100NB(theta):
    word_T100 = []
    t_rank = theta[:]
    len_left = 100
    for item in set(sorted(t_rank, reverse = True)[:100]):
        word_T100.append([lis[list_duplicates_of(theta, item)[i]] for i in range(len(list_duplicates_of(theta, item)) if (len(list_duplicates_of(theta, item)) < 100) else 100)])
        len_left = 100-len(list_duplicates_of(theta, item)) if (len(list_duplicates_of(theta, item)) < 10) else 0    
    return word_T100

'''The following are the scripts to run part6'''
word_T100_L = findTop100L(sess.run(W0), k_vector)
targets = ['positive', 'negative']
for w in range(len(targets)):
    res = 'The top 100 theta that obtained using Naive Bayes for ' + targets[w] + ' review are '
    for i in range(len(word_T100_L[w])-1):
        res += word_T100_L[w][i] + ', '
    print(res+ word_T100_L[w][-1] +'.')
