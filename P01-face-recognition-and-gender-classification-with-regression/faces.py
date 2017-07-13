from numpy import *
from numpy.linalg import norm

#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import os, os.path
from scipy.misc import imsave


act =['Bill Hader', 'Steve Carell']

def formSet(act, trSetSz, validSetSz, testSetSz):
    dataset = []
    trainingSet = []
    validSet = []
    testSet = []
    n = 32 # features in 1D
    i = 0
    # Part 2: Dataset Seperation
    for a in act:
        name = a.split()[1].lower()
        set = os.listdir('.')
        list = [item for item in set if name in item]
        
        dataset.append(list) #output a list where contains all the images has 'name' - last name of the actor
        random.shuffle(dataset[i]) #randomized dataset 
        # Dividing [:100] Training, []
        l = len(dataset[i])
    
        trainingSet.append(dataset[i][:trSetSz])
        validSet.append(dataset[i][(l-validSetSz):]) 
        testSet.append(dataset[i][(l-2*testSetSz):(l-testSetSz)]) 
        i = i+1
    return trainingSet,validSet,testSet
    
  
# Part 3: Classifier of Linear Regression: Bill Hader v.s. Steve Carell
# Logistic Regression Bill - o : Steve Carell - 1
#def sigmoid(z):
#   return 1/(1 - math.exp(-z))
    

# def f(x, y, theta):
#     #x = numpy.vstack( (ones((1, shape(x)[1])), x))
#     #print(shape(x))
#     s = shape(x)[0]
#     #print(shape(x))
#     #print(shape(x), s)
#     x = vstack( (ones((1, s)), x) )
#     #htheta = sigmoid(dot(x,transpose(theta)))
#     return sum(-y * math.log(htheta) - (1-y) * math.log(1-htheta))/m


# def f(x, y, theta):
#     x = vstack( (ones((1, x.shape[1])), x))
#     return sum( (y - dot(x, theta.T)) ** 2)
# 
# def df(x, y, theta):
#     x = vstack( (ones((1, x.shape[1])), x))
#     return -2*sum((y-dot(x, theta.T))*x, 1) 

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)  
    


def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print f(x, y, t)
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n "
        iter += 1
    return t
    
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
    
def readingSet(set):
    x = []
    for a in set:
        print a 
        i = 0 # i = 0 act[0] = 'Bill Hader'
              # i = 1 act[1] = 'Steve Carell'
        for im in a:
            j = 0
            # read each img in trainingSet into 2d array 
            temp = imread(im, True, 'L')
            temp = temp.flatten() 
    
            # Normalized for the images
            temp = temp - 127
            temp = temp /255.0
            
            # append x1 up to x1024
            x.append(temp)
            j += 1
        i += 1
    
    # vstack them
    x = vstack(x)
    return x
    
def trainModel(set): # take set the nest list of images
    tot = size(set)
    pivot = tot/2
    # Definition of the Output
    y = np.append(zeros(pivot),ones(pivot))
    initTheta = 0.0 * ones((32*32+1,))
    initTheta[:] = np.random.rand()
    alpha = 0.00015
    x = readingSet(set)
    # vstack y
    y = vstack(y)
    theta_desired = grad_descent(f, df, x.T, y.T, initTheta, alpha)
    return theta_desired

def performanceEvaluation(theta_desired, input):
    tot = input.shape[0] # tot = sample size
    index_list = range(tot)
    pivot = tot/2

    # first half of set = hader
    succ = 0
    # for hader
    for index in index_list[:pivot]:
        res = sum(theta_desired[1:]*input[index,:])+theta_desired[0]
        if res < 0.5:
            succ += 1
    
    # for carell
    for index in index_list[pivot:]:
        res = sum(theta_desired[1:]*input[index,:])+theta_desired[0]
        print res
        if res > 0.5:
            succ += 1

    res = float(succ)/tot
    return res
    
    
def displayArray(arr):
    # reshape -> 32 * 32
    im = arr[1:].reshape([32, 32])
    # display 
    implot = plt.imshow(im) # To show the impage 
    plt.show()

def partIIIscript():
    act =['Bill Hader', 'Steve Carell']
    #trSetSz, validSetSz, testSetSz = 100, 10, 10
    set1, set2, set3 = formSet(act, 100, 10, 10)
    # train models using trainingSet
    x1 = readingSet(set1)
    x2 = readingSet(set2)
    x3 = readingSet(set3) 
    
    theta_desired = trainModel(set1)
    
    # performance on the validation set
    res1 = performanceEvaluation(theta_desired, x1)
    res2 = performanceEvaluation(theta_desired, x2)
    res3 = performanceEvaluation(theta_desired, x3)
    print res1, res2, res3
    
    
    
        
    
    
##  Part V  
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)  
    
def formSet(act, trSetSz, validSetSz, testSetSz):
    dataset = []
    trainingSet = []
    validSet = []
    testSet = []
    n = 32 # features in 1D
    i = 0
    # Part 2: Dataset Seperation
    for a in act:
        name = a.split()[1].lower()
        set = os.listdir('.')
        list = [item for item in set if name in item]
        
        dataset.append(list) #output a list where contains all the images has 'name' - last name of the actor
        random.shuffle(dataset[i]) #randomized dataset 
        # Dividing [:100] Training, []
        l = len(dataset[i])
    
        trainingSet.append(dataset[i][:trSetSz])
        validSet.append(dataset[i][(l-validSetSz):]) 
        testSet.append(dataset[i][(l-2*testSetSz):(l-testSetSz)]) 
        i = i+1
    return trainingSet,validSet,testSet

def performanceEvaluation(theta_desired, input): 
    tot = input.shape[0] # tot = sample size
    index_list = range(tot)
    print tot
    pivot = tot/2

    # first half of set = hader
    succ = 0
    # for hader
    for index in index_list[:pivot]:
        res = sum(theta_desired[1:]*input[index,:])+theta_desired[0]
        if res < 0.5:
            succ += 1
    
    # for carell
    for index in index_list[pivot:]:
        res = sum(theta_desired[1:]*input[index,:])+theta_desired[0]
        print res
        if res > 0.5:
            succ += 1

    res = float(succ)/tot
    return res
    
def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print f(x, y, t)
            #print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            #print "Gradient: ", df(x, y, t), "\n "
        iter += 1
    return t

szList = [10,15,20,10,15,20]

def partV(act, szList):
    j = 0 
    training = []
    valid = []
    test = []
    
    while j < 6: # Go through the desired actors, seperate them into 2 
        initSz = szList[j] 
        trSet, valSet, ttSet = formSet([(act[j])], initSz*5, initSz*2, initSz)
        training = training + trSet[0]
        valid = valid + valSet[0]
        test = test + ttSet[0]
        j += 1
        
    return training, valid, test
    
def partVread(set): 
    '''return dim(x) = m * n
    '''
    x = ones(0)
    
    for a in set:
        # read each img in trainingSet into 2d array 
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

def partVTrainModel(x, pivot): 
    # x is the 2D matrix 
    # pivot is the size of the output
    # szList = [10,15,20,10,15,20]
    # trSet = initSz*5, valSet = initSz*2, ttSet = initSz
    # Remember 
    # Definition of the Output
    y = np.append(zeros(pivot),ones(pivot)) 
    
    initTheta = 0.0 * ones((32*32+1,))
    initTheta[:] = np.random.rand()
   
    alpha = 0.0000019 #overfitting
   
    # vstack y
    # y = vstack(y)
    
    theta_desired = grad_descent(f, df, x.T, y.T, initTheta, alpha)
    return theta_desired

act_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon','Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
p = 225
def partVFinal(act, act_test, p):
    szList = [10,15,20,10,15,20]
    set = partV(act, szList)
    x = partVread(set[0])
    theta = partVTrainModel(x, p)
    res = performanceEvaluation(theta, x)
    # didnt download the images
    set1 = partV(act_test, szList)
    x1 = partVread(set1[0])
    res1 = performanceEvaluation(theta, x1)
    print res, res1
    
## Part VI c)

def partVIcCost(x, y, theta):
    '''same input dim as before
    x = m * n,  y = k * k, theta = n * k'''
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( sum( (y - dot(theta.T,x)) ** 2) )
    
def partVIcGradident(x, y, theta):
    '''input theta: n * k
    x: n * m, y: k * m
    Output n*k'''
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*( dot(x, (y-dot(theta.T, x)).T ))
    
def partVIdGrad(x, y, theta):
    ''' return grad dim = n * k
    '''
    EPS = 1e-5   #EPS = 10**(-5)
    h = EPS      # init h
    grad = zeros((theta.shape)) #construct a n*k matrix
    for i in range(theta.shape[0]):
        for j in range(theta.shape[1]):
            l = partVIcCost(x, y, theta)
            theta[i][j] += h
            grad = partVIcCost(x, y, theta) - J0
            grad[i][j] = float(grad)/h
    return grad

# Part VII
# form set of 6*6
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']


def partVIIReadSet(act):
    '''initSz = 10
    trSet = initSz*5, valSet = initSz*2, ttSet = initSz'''
    szList = [10, 10, 10, 10, 10, 10]
    trSet, valSet, ttSet = partV(act, szList)
    x_tr = partVread(trSet)
    x_val = partVread(valSet)
    x_tt = partVread(ttSet)
    return x_tr, x_val, x_tt
# problem doesn't return valid list 
   
def partVIIY(x, sampleSz):
    ''' return dim(y) = m * k
    for act[0] y[:m, 0] = 1 and zeros the rest
    '''
    k = 6 # number of labels
    m = sampleSz
    for i in range(k): # x.shape[0] = m
        temp = zeros((m, k))
        temp[:,i] = ones((m,))
        if i == 0:
            y = temp
        else:
            y = vstack((y, temp))
    return y 
       
def partVTrainModel(x, y): 
    '''x is the 2D matrix 
    pivot is the size of the output
    szList = [10,15,20,10,15,20]
    trSet = initSz*5, valSet = initSz*2, ttSet = initSz
    Remember 
    Definition of the Output'''    
    initTheta = 0.0 * ones((32*32+1, 6))
    initTheta[:] = np.random.rand()
    alpha = 0.00000009
    
    theta_desired = grad_descent(partVIcCost, partVIcGradident, x.T, y.T, initTheta, alpha)
    return theta_desired
    
def translation(row):
    '''take each row vector of h dim = 1 * k
    return the value based on the hypothesis threadhold'''
    for i in range(len(row)):
        row[i] = (1 if (row[i] > 0.5) else 0)
    return row

def VIIperformanceEval(theta, input, y):
    k = 6
    tot = input.shape[0] # tot = sample space
    pivot = tot/k # sample size
    score = []
    
    for i in range(6):
        succ = 0
        front = i*pivot
        end = (i+1)*pivot
        h = dot((theta.T)[:,1:],input.T)+theta[i][0]*ones((tot,)) # dim = tot * k
        h = h.T # dim = tot * k
        section = h[front:end, :] # dim = pivot * k
        
        for j in range(pivot): # tranverse the rows 0:49
            row = translation(section[j])
            out = zeros((k,))
            out[i] = 1
            # check equality
            if np.array_equal(out, row):
                succ += 1
        print succ
        score.append(float(succ)/pivot)
                
    return score        

def partVIIfinal(act):
    x, x1, x2 = partVIIReadSet(act) # x <- training; x1 <- valSet; x2 <- ttSet
    y = partVIIY(x, x.shape[0]/6)
    y1 = partVIIY(x1, x1.shape[0]/6)
    y2 = partVIIY(x2, x2.shape[0]/6)
    
    theta = partVTrainModel(x, y)
    res = VIIperformanceEval(theta, x, y)
    res1 = VIIperformanceEval(theta, x1, y2)
    res2 = VIIperformanceEval(theta, x2, y2)
    return res, res1, res2    

def partVIII(theta, act):
    i = 5
    name = act[i].split()[1].lower()
    # reshape -> 32 * 32
    print name
    im = (theta.T)[i][1:].reshape([32, 32])
    # display 
    implot = plt.imshow(im) # To show the impage 
    plt.show()
        
        ## Download imgs
        

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import glob
import scipy.misc
from scipy import ndimage
from numpy import *
from matplotlib.pyplot import *
from scipy.misc import imread
#from scipy.misc import imsave

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imghdr



#act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
act =['Steve Carell']

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.
# 
# def isGray(rgb):
#     if 
#     return True    
def cropAndProcess(img, coords):
    '''return the cropped, grayscaled and resized version of img 
    Input: 
        img is the file, i.e. 'baldwin0.jpg'
        coords: the format of the bounding box, as a string array, len = 1
    '''
    coord = coords.split(',',4)
    x1 = int(coord[0])
    y1 = int(coord[1])
    x2 = int(coord[2])
    y2 = int(coord[3])
    #im = mpimg.imread(img_name) # Image reading 

    
    # print(im.shape)
    im = img[y1:y2, x1:x2] # to heads
    # if !(isGray(im) )
    #     im = rgb2gray(im) # to Grayscalable
    im = imresize(im,(32,32))
    return im 

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #testfile.retrieve(line.split()[4], filename)
            #timeout is used to stop downloading images which take too long to download
            # timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            # if not os.path.isfile("uncropped/"+filename):
            timeout(testfile.retrieve, (line.split()[4], filename), {}, 30)
            if not os.path.isfile(filename):
                continue
            if imghdr.what(filename) == 'jpeg':
                coords = line.split()[5]
                im = imread(filename, True, 'L')
                im = cropAndProcess(im, coords)
        #implot = plt.imshow(im) # To show the impage 
        #plt.show()
                #filename = 'new-'+img
                imsave(filename,im, cmap=cm)
                #print(img)
            
                print filename
                i += 1
    
        
   
