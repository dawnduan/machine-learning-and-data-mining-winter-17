import numpy as np
import time
import pickle as cPickle
import math
import os
from scipy import spatial

M = np.load("embeddings.npz")["emb"]# TODO need trained embeddings
ind = np.load("embeddings.npz")["word2ind"].flatten()[0]#word indices: len-41524

def cosine(a, b): # compute cosine similarity
    return (1 - spatial.distance.cosine(a, b))

def part8(M, ind, word): #Top 10 words using cosine similarity
    ind_desired = list(ind.keys())[list(ind.values()).index(word)] #the index of target word
    cos_val = [cosine(M[i,:], M[ind_desired,:]) for i in range(M.shape[0]) if i != ind_desired] # using cosine
    temp = cos_val[:] # make a copy of the lis and sort the cosine values
    return [ind[cos_val.index(item)] for item in sorted(temp, reverse = True)[:10]]# Top 10 word list

def findfun(a, b): # return embeddings of the sum of two words
    ind_a = list(ind.keys())[list(ind.values()).index(a)] 
    ind_b = list(ind.keys())[list(ind.values()).index(b)]
    return M[ind_a, :] + M[ind_b, :]
    
def checksimi(a): # returm top3 similar words for the input embeddings
    cos_val = [cosine(M[i,:], a) for i in range(M.shape[0])] # using cosine
    temp = cos_val[:]
    return [ind[cos_val.index(item)] for item in sorted(temp, reverse = True)[:3]]# Top3 word list

def checkeql(a,b):
    l = checksimi(findfun(a, b)).remove(a)
    l.remove(b)
    return l # return most similar word for sum(a. b) of their embeddings

'''The following are the scripts to run part8'''
targets = ['story', 'good']
for w in targets:
    res = 'The most similar words to ' + w + ' are '
    for i in range(len(part8(M, ind, w))-1):
        res += part8(M, ind, w)[i] + ', '
    print(res+ part8(M, ind, w)[-1] +'.')
print('word2vec('+checkeql('joy','happy')[0]+') = word2vec(joy) + word2vec(happy)')
print('word2vec('+checkeql('better','good')[0]+') = word2vec(better) + word2vec(good)')

