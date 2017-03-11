'''

This file pre-trains a two-layer LSTM network  to find out the underlying feature representation
of ECG signals.
'''

import pickle
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers.core import Dense, Dropout, Activation, Masking
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import scipy.io as sio
from keras.layers import Merge
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.regularizers import l1, l2
import random
from train_mit import  find_weight
np.random.seed(1000)  # for reproducibility
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Droupout
from keras.layers.recurrent import LSTM
from keras import backend as K
from tsne import tsne
import matplotlib.plot as plt
import matplotlib


def build_model():
    model = Sequential()
    model.add(LSTM(5, 300, return_sequences=True))
    model.add(LSTM(300, 500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, 200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(200, 3))
    model.add(Activation("linear")) model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model

def get_feature(model, layer_num,input):
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[layer_num].output])
    layer_output = get_3rd_layer_output([input])[0]
    return layer_output


class Stopping(Callback):

    def __init__(self, threshold=0.2, epoches=50):
        super(Callback, self).__init__()
        self.threshold = threshold
        self.monitor = 'acc'
        self.epoches = epoches
        self.value =0
        self.patience =0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current > self.threshold or epoch > self.epoches:
            self.value = 1
            self.patience = self.patience +1
            if self.patience >3:
               self.model.stop_training = True
def train_rnn_model(model, train_data, train_label,weight_path):
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
    checkpointers = ModelCheckpoint(filepath=weight_path +'weight.{epoch:02d}-{acc:.4f}.hdf5',
                                    monitor='val_acc', save_best_only=False, mode='auto')
    stopping = Stopping(threshold=0.98, epoches=400)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
    model.fit(train_data, train_label, nb_epoch= 1000, batch_size=128,  shuffle=True, callbacks= [checkpointers,stopping])

def test_rnn_model(model, test_data, test_label, weight_path):
    filename = find_weight(weight_path)
    print(filename)
    model.load_weights(weight_path+filename)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
    prediction = model.predict(test_data, batch_size=128)
    return prediction,

def load_data(data, n_prev=10):
        """
        data should be pd.DataFrame()
        """

        docX, docY = [], []
        for i in range(len(data) - n_prev):
            docX.append(data.iloc[i:i + n_prev].as_matrix())
            docY.append(data.iloc[i + n_prev].as_matrix())
        alsX = np.array(docX)
        alsY = np.array(docY)

        return alsX, alsY


def rnn_main():
     train_data_path = './train_data/'
     test_data_path = './test_data/'
     weight_path = './weight/'
     X_train, y_train= load_data(train_data_path,10)
     X_test, y_test = load_data(test_data_path, 10)
     model = build_model()
     train_rnn_model(model, X_train,y_train,weight_path)
     prediction = test_rnn_model(model, X_test, y_test,weight_path)
     train_feature, train_feature_tsne = feature_visualization(model,4, X_train)
     test_feature, test_feature_tsne = feature_visualization(model, 4, X_test)



def plot_scatter(feature, labels, legends,title):

    x = feature[:,0].tolist()
    y = feature[:,1].tolist()
    label = labels.tolist()
    colors_ori = ['red','green','blue','purple']
    colors=colors_ori[:max(labels)+1]

    fig = plt.figure()
    plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    cb.set_ticklabels(legends[:max(labels)+1])
    plt.title(title)
    plt.show()


    '''
    Plot.scatter(rnn_mor_feature_tsne[:,0], rnn_mor_feature_tsne[:,1], 20, labels)
    Plot.show()
    '''
def feature_visualization(model, layer_num, input):

     feature = get_feature(model, layer_num, input)
     feature_tsne =  tsne(feature, 2, feature.shape[1], 20)
     plot_scatter(feature_tsne[0],0, ['Abnormal', 'Normal'],'Time-feature from Fully-connected Neural Networks')
     return feature, feature_tsne

if __name__ =='__main__':
    rnn_main()





