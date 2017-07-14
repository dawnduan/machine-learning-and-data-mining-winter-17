#!/usr/bin/env python3

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pylab import *
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *
import sys

def wei_update(wei, weights):
    wei['l_pos'].append(weights[0,0])
    wei['l_vel'].append(weights[1,0])
    wei['l_ang'].append(weights[2,0])
    wei['l_rot'].append(weights[3,0])
    
    wei['r_pos'].append(weights[0,1])
    wei['r_vel'].append(weights[1,1])
    wei['r_ang'].append(weights[2,1])
    wei['r_rot'].append(weights[3,1])
  
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()


# Importing the reinforment task 
env = gym.make('CartPole-v0')

RNG_SEED=1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 0.0001
TINY = 1e-8
gamma = 0.99

# Initialize weights and relu 
weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

if args.load_model:
    model = np.load(args.load_model)

w_init = weights_init
b_init = relu_init

# Get action state set size
try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

# Gets observation space set size
Y_UNITS = 1
input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.float32, shape=(None, 2), name='y')

# Obtain probabilities of taking action left and right
dir_probabilities = fully_connected(
    inputs=x,
    num_outputs=output_units,
    activation_fn=tf.nn.softmax,
    weights_initializer=w_init,
    weights_regularizer=None,
    biases_initializer=b_init,
    scope='dir_probabilities')

all_vars = tf.global_variables()

# Define policy function to be normally distributed about mus and sigmas
pi = tf.contrib.distributions.Bernoulli(p=dir_probabilities, name="pi")

# Pass policy function through tanh layer and get the log probability of each action 
pi_sample = pi.sample() 
log_pi = pi.log_prob(y, name='log_pi')

# Define returns to be ... 
Returns = tf.placeholder(tf.float32, name='Returns')

# Run gradient decent to maximize return * log_pi with learning rate alpha
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=25
MAX_STEPS = 200
track_timeStep = [] 
track_returns = []
track_weights = []
wei = {}
wei['l_pos'] = []
wei['l_vel'] = []
wei['l_ang'] = []
wei['l_rot'] = []

wei['r_pos'] = []
wei['r_vel'] = []
wei['r_ang'] = []
wei['r_rot'] = []
for ep in range(16384):
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        #env.render()

        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0])
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break
    
    if not args.load_model:
        print ("Ran here")
        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.vstack(np.array(ep_actions)),
                                Returns:returns })

    
    track_returns.append(G)
    
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    track_timeStep.append(mean_return)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
 
   
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))

    print("Episode AVERAGE so far {}".format(sum(track_timeStep)/len(track_timeStep)))
    with tf.variable_scope("dir_probabilities", reuse=True):
        print("incoming weights for the dir_probabilities's from the input unit/state:", sess.run(tf.get_variable("weights"))[0,:])
        track_weights.append(sess.run(tf.get_variable("weights")))
        
        wei_update(wei, sess.run(tf.get_variable("weights")) )
        
    if (sum(track_timeStep)/len(track_timeStep) > 70): 
        break
sess.close()



##x = np.arange(0., num_iter) # num_iter
x = [i for i in range(len(track_weights))]
#pos = plt.plot(x, wei['l_pos'],  'g--', label="Left position")
#vel = plt.plot(x, wei['l_vel'], 'r--', label="Left velocity")
#ang = plt.plot(x, wei['l_ang'], 'b--',label="Left angle")
#rot = plt.plot(x, wei['l_rot'], 'c--', label="Left rotation")

#r_pos = plt.plot(x, wei['r_pos'],  'g', label="Right position")
#r_vel = plt.plot(x, wei['r_vel'], 'r', label="Right velocity")
#r_ang = plt.plot(x, wei['r_ang'], 'b',label="Right angle")
#r_rot = plt.plot(x, wei['r_rot'],'c', label="Right rotation")

time_s = plt.plot(x, track_timeStep,'c--', label="Avg number of time-step")
plt.axis([0, len(track_weights), 0, 100])

plt.legend()#[pos, vel, ang, rot], ["pos", "vel", "ang", "rot"])
plt.show()
