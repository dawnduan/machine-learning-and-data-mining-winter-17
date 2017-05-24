\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{listings}
\documentclass[12pt]{article}

\usepackage[utf8]{inputenc}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{tabu}
\graphicspath{ {images/} }
\usepackage{subcaption}
\usepackage{minted}
\usepackage{xcolor}
\usepackage{amssymb}


\title{Deep Neural Networks for Handwritten Digit and Face Recognition}
\date{March 2017}

\begin{document}

\maketitle

\section{Part1}
Data set include training data(train number) and testing data(test number) from digit 0 to digit 9.  
\\
\includegraphics[scale=1]{train0_rand0.jpg}
\includegraphics[scale=1]{train0_rand1.jpg}
\includegraphics[scale=1]{train0_rand2.jpg}
\includegraphics[scale=1]{train0_rand3.jpg}
\includegraphics[scale=1]{train0_rand4.jpg}
\includegraphics[scale=1]{train0_rand5.jpg}
\includegraphics[scale=1]{train0_rand6.jpg}
\includegraphics[scale=1]{train0_rand7.jpg}
\includegraphics[scale=1]{train0_rand8.jpg}
\includegraphics[scale=1]{train0_rand9.jpg}

\includegraphics[scale=1]{train1_rand0.jpg}
\includegraphics[scale=1]{train1_rand1.jpg}
\includegraphics[scale=1]{train1_rand2.jpg}
\includegraphics[scale=1]{train1_rand3.jpg}
\includegraphics[scale=1]{train1_rand4.jpg}
\includegraphics[scale=1]{train1_rand5.jpg}
\includegraphics[scale=1]{train1_rand6.jpg}
\includegraphics[scale=1]{train1_rand7.jpg}
\includegraphics[scale=1]{train1_rand8.jpg}
\includegraphics[scale=1]{train1_rand9.jpg}

\includegraphics[scale=1]{train2_rand0.jpg}
\includegraphics[scale=1]{train2_rand1.jpg}
\includegraphics[scale=1]{train2_rand2.jpg}
\includegraphics[scale=1]{train2_rand3.jpg}
\includegraphics[scale=1]{train2_rand4.jpg}
\includegraphics[scale=1]{train2_rand5.jpg}
\includegraphics[scale=1]{train2_rand6.jpg}
\includegraphics[scale=1]{train2_rand7.jpg}
\includegraphics[scale=1]{train2_rand8.jpg}
\includegraphics[scale=1]{train2_rand9.jpg}

\includegraphics[scale=1]{train3_rand0.jpg}
\includegraphics[scale=1]{train3_rand1.jpg}
\includegraphics[scale=1]{train3_rand2.jpg}
\includegraphics[scale=1]{train3_rand3.jpg}
\includegraphics[scale=1]{train3_rand4.jpg}
\includegraphics[scale=1]{train3_rand5.jpg}
\includegraphics[scale=1]{train3_rand6.jpg}
\includegraphics[scale=1]{train3_rand7.jpg}
\includegraphics[scale=1]{train3_rand8.jpg}
\includegraphics[scale=1]{train3_rand9.jpg}

\includegraphics[scale=1]{train4_rand0.jpg}
\includegraphics[scale=1]{train4_rand1.jpg}
\includegraphics[scale=1]{train4_rand2.jpg}
\includegraphics[scale=1]{train4_rand3.jpg}
\includegraphics[scale=1]{train4_rand4.jpg}
\includegraphics[scale=1]{train4_rand5.jpg}
\includegraphics[scale=1]{train4_rand6.jpg}
\includegraphics[scale=1]{train4_rand7.jpg}
\includegraphics[scale=1]{train4_rand8.jpg}
\includegraphics[scale=1]{train4_rand9.jpg}

\includegraphics[scale=1]{train5_rand0.jpg}
\includegraphics[scale=1]{train5_rand1.jpg}
\includegraphics[scale=1]{train5_rand2.jpg}
\includegraphics[scale=1]{train5_rand3.jpg}
\includegraphics[scale=1]{train5_rand4.jpg}
\includegraphics[scale=1]{train5_rand5.jpg}
\includegraphics[scale=1]{train5_rand6.jpg}
\includegraphics[scale=1]{train5_rand7.jpg}
\includegraphics[scale=1]{train5_rand8.jpg}
\includegraphics[scale=1]{train5_rand9.jpg}

\includegraphics[scale=1]{train6_rand0.jpg}
\includegraphics[scale=1]{train6_rand1.jpg}
\includegraphics[scale=1]{train6_rand2.jpg}
\includegraphics[scale=1]{train6_rand3.jpg}
\includegraphics[scale=1]{train6_rand4.jpg}
\includegraphics[scale=1]{train6_rand5.jpg}
\includegraphics[scale=1]{train6_rand6.jpg}
\includegraphics[scale=1]{train6_rand7.jpg}
\includegraphics[scale=1]{train6_rand8.jpg}
\includegraphics[scale=1]{train6_rand9.jpg}

\includegraphics[scale=1]{train7_rand0.jpg}
\includegraphics[scale=1]{train7_rand1.jpg}
\includegraphics[scale=1]{train7_rand2.jpg}
\includegraphics[scale=1]{train7_rand3.jpg}
\includegraphics[scale=1]{train7_rand4.jpg}
\includegraphics[scale=1]{train7_rand5.jpg}
\includegraphics[scale=1]{train7_rand6.jpg}
\includegraphics[scale=1]{train7_rand7.jpg}
\includegraphics[scale=1]{train7_rand8.jpg}
\includegraphics[scale=1]{train7_rand9.jpg}

\includegraphics[scale=1]{train8_rand0.jpg}
\includegraphics[scale=1]{train8_rand1.jpg}
\includegraphics[scale=1]{train8_rand2.jpg}
\includegraphics[scale=1]{train8_rand3.jpg}
\includegraphics[scale=1]{train8_rand4.jpg}
\includegraphics[scale=1]{train8_rand5.jpg}
\includegraphics[scale=1]{train8_rand6.jpg}
\includegraphics[scale=1]{train8_rand7.jpg}
\includegraphics[scale=1]{train8_rand8.jpg}
\includegraphics[scale=1]{train8_rand9.jpg}

\includegraphics[scale=1]{train9_rand0.jpg}
\includegraphics[scale=1]{train9_rand1.jpg}
\includegraphics[scale=1]{train9_rand2.jpg}
\includegraphics[scale=1]{train9_rand3.jpg}
\includegraphics[scale=1]{train9_rand4.jpg}
\includegraphics[scale=1]{train9_rand5.jpg}
\includegraphics[scale=1]{train9_rand6.jpg}
\includegraphics[scale=1]{train9_rand7.jpg}
\includegraphics[scale=1]{train9_rand8.jpg}
\includegraphics[scale=1]{train9_rand9.jpg}
\section{Part2}
Implementation as follow:
\begin{minted}{python}
#Return the dot product between x(images) and w(weight) and then add b(bias).
def net_w(x, w):
    b = ones((1000, 10)) 
    return (dot(x, w) + b).T 

#script to demo:
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
\end{minted}
\section{Part3}
3(a)\\
\includegraphics[scale=0.5]{3(a).jpg}
3(b)\\
Implementation includes cost(x,y,w), dcost(x,y,w), grad_descent(f, df, x, y, init_t, alpha) and finiteDiff(x, y, theta).

\begin{lstlisting}[language=Python]
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
       
    #Script to demo: 
    x = [] #sizeoft * 784
    y = [] #10 * sizeoft
    w = [] #shape (784,10) .
    w_temp = np.ones((28*28,)) 
    w_temp[:] = np.random.rand()
    for i in range(10): # every label
        x_temp = M["train"+str(i)][:25,:]/255.0
        y_temp = zeros((25,10))
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
\end{lstlisting} 
\section{Part4}
Implementation includes performance(x, y , w, sizeofdata) and batch(datasize).\\
4(a)\\Graph of learning curves is as follow: The error rate will decrease when iteration increase. In terms of the training set, error rate will come close to 0 when the iteration is extremely large. For the testing set, the error rate will keep decreasing but will not reach 0.
The blue line is error rate in training set and the green line is error rate in testing set.\\
\includegraphics[scale=0.5]{4(a).png}
4(b)
\includegraphics[scale=0.4]{4(b).png}
4(c)
The optimization procedure used for this question is as follows: Instead of training on all the data at once, we split the training data into several sets and periodically checked the performance to try and detect overfitting. When overfitting was detected, we discarded the training process and started over. We use the batch function to generate training data of a specified size:
\begin{lstlisting}[language=Python]
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
\end{lstlisting} 
\section{Part5}
Pick 90 images for each digit, and generate another 10 images for each digit as noise. For those "noise" images, we mean that if the number actually is 9, treat it as 4. If it is 1, treat it as 6. These 100 bad images will act as noise in the training data and it will effect the weight that is generated by gradient descent.\\
In the beginning, the accuracy of linear regression(86.8\%) is higher than logistic regression(91.8\%) on the training set. However on the test set, the accuracy of logistic regression(81.4\%) is higher than linear regression(71.4\%).\\
The reasons for this is because in linear regression, each image will have same effect on theta,however, in logistic regression, each image may or may not have same effect on theta. e.g linear regression is a line.one random point would shift the line a lot. But for logistic regression, one random noise will not have big effect on total weight.
\begin{lstlisting}[language=Python]
    #script to demo
    #Part 4 has testing data
    noise_x = [] #sizeoft * 784
    noise_y = [] #10 * sizeoft
    noise_w = [] #shape (784,10) .
    w_temp = np.ones((28*28,)) 
    w_temp[:] = np.random.rand()
    for i in range(10): 
        x_temp = M["train"+str(i)][:90,:]/255.0 
        y_temp = zeros((90,10))
        y_temp[:,i] = 1
        if i == 0:
            noise_x = x_temp
            noise_y = y_temp
            noise_w = w_temp    
        else:
            noise_x = vstack((noise_x,x_temp))
            noise_y = vstack((noise_y,y_temp))
            noise_w = vstack((noise_w,w_temp))
        x_temp = M["train"+str(i)][90:100,:]/255.0 
        y_temp = zeros((10,10))
        y_temp[:,i-5] = 1
        noise_x = vstack((noise_x,x_temp))
        noise_y = vstack((noise_y,y_temp))
    noise_w = noise_w.T
    
    w_logistireg = grad_descent(cost, dcost, noise_x, noise_y, noise_w, alpha) 
    w_lineareg = grad_descent(f, df, noise_x, noise_y, noise_w, alpha)
    print("Using bad tranining data set to training")
    print("logisti Regression")
    print performance(noise_x,noise_y,w_logistireg, 1000) 
    print("linear Regression")
    print performance(noise_x,noise_y,w_lineareg, 1000)
    
    print("Testing data")
    print("logistic Regression ")
    print performance(testx,testy,w_logistireg, 1000)  
    print("linear Regression")
    print performance(testx,testy,w_lineareg, 1000) 
\end{lstlisting} 
\section{Part6}
\includegraphics[scale=0.2]{6(a).jpg}


\section{Part7}
The learning curve for the test, training, and validation sets has been shown below. The x-axis is the number of iterations and y-axis is error rate. The learning curve is shown below. Testing set has 83.8889\% accuracy, validation set has 75\% accuracy, training set has 100\% accuracy which means penalty(error) is 0.
\\
\includegraphics[scale=0.45]{p07.png}

The final performance classification on the test set is about 78\% of accuracy. The text description of our system is as followed. In particular, the input is preprocess by the following steps. Firstly, all the images has been re-downloaded, cropped and resized used the same function from A01 (included) except the following lines. Particularly, the follwoing codes are used to remove non-faces from our imgae dataset, utilized the SHA-256 hashes to remove bad images. 
\begin{lstlisting}[language=Python]
import hashlib

m = hashlib.sha256()
m.update(open("uncropped/"+filename, 'rb').read())
if m.hexdigest() != line.split()[6]:
    os.remove("uncropped/"+filename)
    print('not match sha256, has removed')
else: 
    print(filename)
    i += 1
\end{lstlisting} 
\\
After preprocessing all the images for each actors to size of 32x32, head-centered,they are ready to be read into flattened numpy array, with size [1024, 1]. There are two main functions used to read every images for actors and categorized them into a dictionary with keys as followed: $['test_drescher', 'train_hader', 'train_carell',\\ 'train_chenoweth', 'test_carell', 'valid_chenoweth', 'valid_drescher', 'valid_carell', 'valid_ferrera', 'test_ferrera',\\ 'test_hader', 'test_chenoweth', 'valid_hader', 'train_baldwin', 'test_baldwin', 'train_ferrera', 'train_drescher',\\ 'valid_baldwin']$. As names suggested, the images are grouped into the training, test, valid sets of each actor by the function formM. The input of formM comes from the output of function readSet which reads all the images as 1D numpy array. \\ \\
The specifications of the the neural networks are as followed. The activation functions of input layer is tanh, which takes the linear combination of x, W0 and the bias units b0. The one hidden unit use the softmax as the activation functions generate outputs as probability. The code snippet is as followed: \\

\begin{lstlisting}[language=Python]
W0 = tf.Variable(tf.random_normal([1024, nhid], stddev=0.1))
b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))
W1 = tf.Variable(tf.random_normal([nhid, 6], stddev=0.01))
b1 = tf.Variable(tf.random_normal([6], stddev=0.01))
layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
layer2 = tf.matmul(layer1, W1)+b1
y = tf.nn.softmax(layer2)
y_ = tf.placeholder(tf.float32, [None, 6])
\end{lstlisting} 
The cost function is shown 
\begin{lstlisting}[language=Python]
lam = 0.0
decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
reg_NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
train_step = tf.train.AdamOptimizer(0.0005).minimize(reg_NLL)
\end{lstlisting} 


\section{Part8}
A scenario where using regularization is necessary in the context of face classification, would be when we experience overfitting. Specifically, a small sized training set, a limited number of features, and over-repeated iterations can all be possible reasons causing overfitting. From the learning curve, overfitting can be observed from the learning curve as well, specifically, when the error rate of the training set stays on zero and that of the test set starts to go up. There should be a clearly global minimum has been presented before, as seen in figure.   

we have 40 pictures for each actor.15000 iterations.
\\
\includegraphics[scale=0.4]{p8ovf02.png}
\includegraphics[scale=0.4]{p8reg02.png}\\
left:before apply regularization
right:after regularization
\\
The best regularization parameter λ is selected as 0.003. This leads to an enhancement of 6 percent roughly, from original error rate of 2.2 percent to 1.6 percent after the weight penalty. The degree of improvement is substantial, given that the size of our test size, N, is 180. The expected should be $(\frac{75}{\sqrt{N}})\% = 5.59\%$.
\section{Part9}
The two of the actors selected are .The visualization the weights of the hidden units that are useful for classifying input photos as those particular actors has been shown below. \\
\includegraphics[scale=0.45]{p09by7_Drescher.png}
\includegraphics[scale=0.45]{p09by7_hader.png}\\
The first image is female actor, Fran Drescher and the second one is male actor, Bill Hader. The selected hidden units are chosen because their inputs, weights from previous layer, have the biggest value across all the weights. This means that the weights are close to their connected inputs the most. In another words these weights are supposed to look like a face the most. The code snippet has included.
\begin{lstlisting}[language=Python]
#Code for displaying a feature from the weight matrix mW
fig = figure(1)
ax = fig.gca()   

#get the index at which the output is the largest
ind_a0 = argmax(sess.run(W1)[:,0])#Fran Drescher
ind_a4 = argmax(sess.run(W1)[:,4])#Bill Hader 

#heatmap = ax.imshow(sess.run(W0)[:,ind_a0].reshape((32,32)), cmap = cm.coolwarm)
heatmap = ax.imshow(sess.run(W0)[:,ind_a4].reshape((32,32)), cmap = cm.coolwarm)
fig.colorbar(heatmap, shrink = 0.5, aspect=5)
show()
\end{lstlisting}
As shown, we first got the index of the biggest value in the W1 ([650x6]) for each actor ([650x1]). That is, the index would be the selected hidden unit that we intend to display, as explained by the reasons above.
\section{Part10}
Extract the values of the activations of AlexNet on the face images has been achieved by the following. By modifying the original alexnet codes provided, we changed the input to a dictionary of numpy array images, with the dimension of [227,227,3]. We substitue into the images using the codes as followed. \\
\begin{lstlisting}[language=Python]
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
\end{lstlisting}
After that, the output from conv4 has the dimension of [, 13, 13, 384]. We flattened these activations into [, 13*13*384] and feed into the same network as accomplished in part7. In other words, we used the activations from alexNet's conv4 layers as our features to our fully connected neural networks. The specifications are as followed. The specifications of the the neural networks are as followed. The activation functions of input layer is tanh, which takes the linear combination of x, W0 and the bias units b0. The one hidden unit use the softmax as the activation functions generate outputs as probability. % see if you can add anything to it


\end{document}