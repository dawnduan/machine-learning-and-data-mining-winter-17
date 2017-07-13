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

embedding = load("embeddings.npz")["emb"]
ind = load("embeddings.npz")["word2ind"].flatten()[0]

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
    
    
## Part 7 code 
def compute_network(): 
    adj_words, dist_words = generateInputs(100, 0)
    val_adj, val_dist = generateInputs(15, 110) 
    test_adj, test_dist = generateInputs(15, 210) 
    valBatch_xs, valBatch_ys = generateBatch(300, val_adj, val_dist)
    testBatch_xs, testBatch_ys = generateBatch(300, test_adj, test_dist)
    
    # Generate network 
    x = tf.placeholder(tf.float32, [None, 256])
    
    W0 = tf.Variable(tf.random_normal([256, 1], stddev=0.5))
    b0 = tf.Variable(tf.random_normal([1], stddev=0.01))
    
    layer1 = tf.matmul(x, W0)+b0
    
    
    y = tf.nn.sigmoid(layer1) 
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    
    lam = 0.0005
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))
    reg_NLL = -tf.reduce_sum((y_*tf.log(y)) + ((1-y_)*tf.log(1-y)))
    
    # 0.0005
    train_step = tf.train.AdamOptimizer(0.00005).minimize(reg_NLL)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #print("Train:", sess.run(accuracy, feed_dict={x: matrix, y_: train_y}))
    
    for i in range(5000):
        #print i  
        batch_xs, batch_ys = generateBatch(100, adj_words, dist_words)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
  
        if i % 20 == 0:
            print "i=",i
            batch_xs, batch_ys = generateBatch(100, adj_words, dist_words)

            print "Train:", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})
            print "Validation:", sess.run(accuracy, feed_dict={x: valBatch_xs, y_: valBatch_ys})
            print "Test:", sess.run(accuracy, feed_dict={x: testBatch_xs, y_: testBatch_ys})
            
        
# Generate list of tuples of adjacent words and list of tuples of distant words
def generateInputs(amnt, offset): 
    story = []
    # Go through as many reviews as specified in amount
    for i in range(amnt): 
        story += M['train_pos'][i + offset]
        
    adj_words = getAdjacent(story)
    print("Finished with adjacent words") 
    
    dist_words = getDistant(story, adj_words) 
    print("Getting distant words")
    
    return adj_words, dist_words
        
# Generates batches of indicated size from list of adjacent words and dist words
def generateBatch(size, adj_words, dist_words): 
    np.random.shuffle(adj_words)
    
    # Create matrix input
    posMatrix = getEmbeddingMatrix(adj_words, size)
    len = posMatrix.shape[0]
    matrix = vstack((posMatrix, getEmbeddingMatrix(dist_words, len)))
    
    # Create Y 
    y = vstack(((np.array([ones(len)])).T, np.array([zeros(len)]).T))
    
    return matrix, y
    
# Generate a list of tuples of adjacent words 
def getAdjacent(list_words): 
    adj_list = []
    
    for i in range(len(list_words) - 1):
        to_append = (list_words[i], list_words[i+1])
        adj_list.append(to_append)
        
    return list(set(adj_list))
    
# Generate a list of tuples of distant words
def getDistant(list_words, positive_words): 
    dist_list = []
    max = len(positive_words)
    
    for i in range(max): 
        # Obtain a random tuple of two words
        rand_tuple = (list_words[randint(0,max)], list_words[randint(0,max)])
    
        # Make sure they are not in adjacent and not already in dist list yet still valid
        while ((rand_tuple in positive_words) or (rand_tuple in dist_list) 
            or (not validTuple(rand_tuple))):
            rand_tuple = (list_words[randint(0,max)], list_words[randint(0,max)])
        
        dist_list.append(rand_tuple)
        
    return dist_list
    
# Turn all tuples into a matrix of their embeddings upto max amount
def getEmbeddingMatrix(tuples_list, max):
    matrix = ones(256)
    count = 0 
    
    # Add tuples up until matrix
    for tup in tuples_list:  
        if (count >= max): 
            break
            
        # See if words have valid embeddeding and if so, add to matrix
        try: 
            index1 = ind.keys()[ind.values().index(tup[0])]
            index2 = ind.keys()[ind.values().index(tup[1])]
        
            matrix = vstack((matrix, np.append(embedding[index1], embedding[index2])))
        except:
            pass
        count += 1
            
    return matrix[1:] 
    
# Check words in tuple exist in dictionary 
def validTuple(tup):
    if ((tup[0] in ind.values()) and (tup[1] in ind.values())): 
        return True
    return False
    

    
