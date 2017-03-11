#!/usr/bin/env python
# coding=utf-8
import cPickle as pickle
import numpy as np
import argparse
import os
import json
#import sys  
import numpy as np  
from sklearn import preprocessing
#from sklearn import feature_extraction    
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
#from sklearn.decomposition import  LatentDirichletAllocation
#from sklearn.feature_extraction.text import HashingVectorizeri
import lda 


def loadData(dataroot,filename):
    file_path=os.path.join(dataroot,filename)
    data=pickle.load(open(file_path,'rb'))
    corpus=[]
    for line in data:
        corpus.append(line[1])
    return corpus
def loadStop(dataroot,filename):
    file_path=os.path.join(dataroot,filename)
    stop_words=[]
    f=open(file_path,'r')
    for line in f:
        stop_words.append(line.strip())
    return stop_words

def tfVector(corpus,stop_words):
    print("Extracting features for:\n")
    
    tf_vectorizer=CountVectorizer(max_df=0.95,min_df=2,
                                 stop_words=stop_words)
    tf=tf_vectorizer.fit_transform(corpus)
    weight=tf.toarray()
    vocab=tf_vectorizer.get_feature_names()
    print 'vocab:{}'.format(vocab)
    return weight,vocab


def toArray(corpus):
    print 'toArray:\n'
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)
    #analyze = vectorizer.build_analyzer()
    vocab = vectorizer.get_feature_names()
    weight = X.toarray()
    print 'vocab:{}'.format(vocab)
    return weight,vocab

def LDA(corpus,weight,vocab,num_topics):
    print 'LDA:\n'
    n_top_words=16
    model=lda.LDA(n_topics=num_topics,n_iter=500,random_state=1)
    model.fit(weight)
    topic_word=model.topic_word_
    print 'topics contain word:\n'
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    doc_topic=model.doc_topic_
    print("type(doc_topic): {}".format(type(doc_topic))) 
    print("shape: {}".format(doc_topic.shape))
    #the doc belong to which topics
    #for i in range(10):
    #    print("{} (top topic: {})".format(corpus[i], doc_topic[i].argmax()))
    #print doc_topic
    features=preprocessing.normalize(doc_topic, norm='l2')
    print features
    return features



def main(params):
    dataroot=params['dataroot']
    filename=params['filename']
    stop_filename=params['stop_filename']
    n_topic=params['n_topic']
    corpus=loadData(dataroot,filename)
    stop_words=loadStop(dataroot,stop_filename)
    weightArray,vocab=tfVector(corpus,stop_words)
    #weightArray,vocab=toArray(corpus)
    blob={}
    features=LDA(corpus,weightArray,vocab,n_topic)
    output_file=os.path.join(dataroot,params['outputfile'])
    blob['features']=features
    pickle.dump(blob,open(output_file,'w'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot',dest='dataroot',default='.')
    parser.add_argument('--filename',dest='filename',default='poll_id_to_title.pklb')
    parser.add_argument('--stop_filename',dest='stop_filename',default='stop_words.txt')
    parser.add_argument('--n_topic',dest='n_topic',default=5)
    parser.add_argument('--outputfile',dest='outputfile',default='features.pkl')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed parameters:'
    print json.dumps(params, indent = 2)
    main(params)

    
