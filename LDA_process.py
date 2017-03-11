#!/usr/bin/env python
# coding=utf-8
import cPickle as pickle
import numpy as np
import argparse
import os
import json
#import sys  
import numpy as np  
#from sklearn import feature_extraction    
#from sklearn.feature_extraction.text import TfidfTransformer    
from sklearn.feature_extraction.text import CountVectorizer  
#from sklearn.feature_extraction.text import HashingVectorizeri
import lda 


def loadData(dataroot,filename):
    file_path=os.path.join(dataroot,filename)
    data=pickle.load(open(file_path,'rb'))
    corpus=[]
    for line in data:
        corpus.append(line[1])
    return corpus

def toArray(corpus):
    print 'toArray:\n'
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)
    #analyze = vectorizer.build_analyzer()
    vocab = vectorizer.get_feature_names()
    weight = X.toarray()
    print weight
    return weight,vocab

def LDA(corpus,weight,vocab,num_topics):
    print 'LDA:\n'
    n_top_words=16
    model=lda.LDA(n_topics=num_topics,n_iter=500,random_state=1)
    model.fit(weight)
    topic_word=model.topic_word_
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    doc_topic=model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic))) 
    print("shape: {}".format(doc_topic.shape))
    for i in range(10):
        print("{} (top topic: {})".format(corpus[i], doc_topic[i].argmax()))


def main(params):
    dataroot=params['dataroot']
    filename=params['filename']
    n_topic=params['n_topic']
    corpus=loadData(dataroot,filename)
    weightArray,vocab=toArray(corpus)
    LDA(corpus,weightArray,vocab,n_topic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',dest='dataroot',default='.')
    parser.add_argument('--filename',dest='filename',default='poll_id_to_title.pklb')
    parser.add_argument('--n_topic',dest='n_topic',default=32)
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed parameters:'
    print json.dumps(params, indent = 2)
    main(params)

    
