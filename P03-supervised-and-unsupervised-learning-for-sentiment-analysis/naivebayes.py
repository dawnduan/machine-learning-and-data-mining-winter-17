import os
import re
import random
import math
import numpy as np
import operator
from functools import partial


###Part 1
M = dict()
trSetSz = 600
validSetSz = 200
testSetSz = 200

stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

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
        # Remove punctuation & lower case the words
        file = open('./review_polarity/txt_sentoken/'+key[-3:]+'/'+film, 'r')
        text = file.read().lower()
        # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
        text = re.sub('[^a-z\ \']+', " ", text)
        temp.append([w for w in list(text.split()) if w not in stopwords])
        file.close()
    M[key] = temp #overwrite the words into dict.values()
    
def part2v1(revList, m, k):
    res = 0
    for rev in revList: #revs per test/valid/trainset
        pos = words_given_label(wordlist(rev), M['train_pos'], m, k)
        neg = words_given_label(wordlist(rev), M['train_neg'], m, k)
        if 'pos' in revList:
            if pos > neg:
                res += 1
        else:
            if pos < neg:
                res += 1
    return float(res)/200.0
    
def wordlist(films): 
    ''' input: nested list; output: all words in the set without duplicated words in each list in the input'''
    dlis = []
    for f in films:
        dlis += f
    return list(set(dlis))
    
def words_given_label(words, films, m, k):
    return [math.log(cond_p(word, films, m, k)) for word in words if len([word for film in films if word in film]) < 30]

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs
    
def cond_p(word, films, m, k):
    return float(len([word for film in films if word in film]) + m*k)/float(200 + k)   

m = 1.1
k = 0.1

def part3():
    theta = []
    for rev in ['pos', 'neg']:
        lis = []
        for key in M.keys():
            if rev in key:
                lis += wordlist(M[key]) # lis contains all the words in pos; a list containing 59338 words
        temp = words_given_label(lis, M['train_'+rev], m, k) # theta values lis
        theta.append(temp)
    return theta, lis

def top_words(temp, wordlis, n):
    '''temp are one nested loop with 2 inner loop: temp[0] is theta for each word
    n is the top n words can be selected
    return one nested loop with 2 inner loop: top_words is the top n words for pos rev'''
    top_words = []
    for j in range(len(['pos', 'neg'])):
        word_T = []
        t_rank = temp[j][:]
        len_left = n
        for item in set(sorted(t_rank, reverse = True)[:n]): # to find n words when possible duplicate theta values
            word_T += [wordlis[list_duplicates_of(temp[j], item)[i]] for i in range(len(list_duplicates_of(temp[j], item)) if (len(list_duplicates_of(temp[j], item)) < n) else n)]
            len_left = n-len(list_duplicates_of(temp[j], item)) if (len(list_duplicates_of(temp[j], item)) < n) else 0    
        top_words.append(word_T)
    return top_words

'''The following are the scripts to run part3'''
# this only prints out the top words nicely with given word list
theta, lis = part3()
word_T10 = top_words(theta, lis, 10)
word_T100 = top_words(theta, lis, 100)
targets = ['positive', 'negative']
for w in range(len(targets)):
    res1 = 'The most influential 10 words for ' + targets[w] + ' review are '
    for i in range(len(word_Tn[w])-1):
        res += word_Tn[w][i] + ', '
    print(res+ word_Tn[w][-1] +'.')
