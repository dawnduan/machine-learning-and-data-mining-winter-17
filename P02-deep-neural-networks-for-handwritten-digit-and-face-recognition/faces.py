from numpy import *
from numpy.linalg import norm

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.pyplot import *
from numpy import *

act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

# Part 2: Dataset Seperation
for a in act:
    name = a.split()[1].lower()
    i = 0
    [item for item in os.listdir('.') if name in item)] 
 ####   
# Part 3: Classifier of Linear Regression: Bill Hader v.s. Steve Carell
# Logistic Regression Bill - o : Steve Carell - 1
def sigmoid(z):
    return 1/(1 - math.exp(-z))
    

def f(x, y, theta):
    x = numpy.vstack( (ones((1, shape(x)[1])), x))
    htheta = sigmoid(dot(x,transpose(theta)))
    
    return sum(-y * math.log(htheta) - (1-y) * math.log(1-htheta))/m

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)   
    

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 500 == 0:
            print "Iter", iter
            print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t)) 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t
    

act = ['Bill Hader', 'Steve Carell']
n = 32
theta = []
x = []
theta0 = [[0. for x in range(n)] for y in range(n)]
y = [[1 for x in range(n)] for y in range(n)]
To Do init space for x and theta
for a in act:
    name = a.split()[1].lower()
    i = 0
    # Matrix(n,n) Init - a list containing n lists, each of n items, all set to 0.1
    # initial theta as 0.1 - arbitrary
    theta.append([[0.1 for x in range(n)] for y in range(n)])
    x.append(imread('img_name', True, 'L'))
    
    for i in range(0, n):
        for j in range(0, n):
            # # hypothesis function
            # h = theta(k)[i,j] * x(k)[i,j]
            # 
            # # cost Fcn
            # diff = h - y(i)[i,j]
            # cost = power(diff, 2)
            # 
            cost += f(x(k)[i,j], y, theta[k][i,j])
            
    
    
    ##### Notes
# Creates a list containing 5 lists, each of 8 items, all set to 0
w, h = 8, 5. 
Matrix = [[0 for x in range(w)] for y in range(h)]

    fsum(...)
        fsum(iterable)
        
        Return an accurate floating point sum of values in the iterable.
        Assumes IEEE-754 floating point arithmetic.
        
    pow(...)
        pow(x, y)
        
        Return x**y (x to the power of y).
        
        
            
def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)    
    

    
    
