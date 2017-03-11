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
from train_mit import  find_weight, build_nn_model, build_rnn_model
np.random.seed(1000)  # for reproducibility



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Droupout
from keras.layers.recurrent import LSTM


model = Sequential()
model.add(LSTM(5, 300, return_sequences=True))
model.add(LSTM(300, 500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(500, 200, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(200, 3))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")



def disorgnize(sequences, nums):
    order = range(len(sequences))
    random.shuffle(order)
    return order[:nums]

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
def train_rnn_model(model_sequence, training_signals, training_premature_flag, training_label, class_weights, rnn_weight_path, weights_previous, iter_num):
    checkpointers = ModelCheckpoint(filepath=rnn_weight_path+'weight.{epoch:02d}-{acc:.4f}.hdf5', monitor='val_acc',
                                    save_best_only=False, mode='auto')
    stopping = Stopping(threshold=0.99, epoches=300)
    model_sequence.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])

    weight = find_weight(weights_previous)
    model_sequence.load_weights(weights_previous + weight)
    model_sequence.fit([training_signals, training_premature_flag], training_label, nb_epoch= 1000,batch_size=128, shuffle=True, callbacks= [checkpointers,stopping],
                         class_weight=class_weights)


def train_nn_model(model_nn, training_time, training_label, class_weights, nn_weight_path, weights_previous, iter_num):
    checkpointers = ModelCheckpoint(filepath=nn_weight_path+'weight.{epoch:02d}-{acc:.4f}.hdf5',
                                    monitor='val_acc', save_best_only=False, mode='auto')
    stopping = Stopping(threshold=0.98, epoches=400)
    model_nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    weight = find_weight(weights_previous)
    model_nn.load_weights(weights_previous + weight)
    model_nn.fit(training_time, training_label, nb_epoch= 1000, batch_size=128,  shuffle=True, callbacks= [checkpointers,stopping],
                  class_weight=class_weights)

def test_nn_model(model, testing_data, testing_label, weight_path):

    filename = find_weight(weight_path)
    model.load_weights(weight_path+filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    prediction = model.predict(testing_data, batch_size=512)
    class_label = 0
    return prediction, 0, class_label


def test_rnn_model(model, testing_signals, testing_premature_flag, testing_label, weight_path):
    filename = find_weight(weight_path)
    print(filename)
    model.load_weights(weight_path+filename)
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
    prediction = model.predict([testing_signals,testing_premature_flag], batch_size=128)
    class_label =0
    #class_label = model.predict_classes([testing_signals,testing_premature_flag], batch_size=512)
    weights_best = 0

    return prediction, weights_best, class_label

def select_data_in_iteration_st(rnn_predictions, nn_predictions, rnn_classes, nn_classes, test_signal):
    sample_num = len(rnn_predictions)
    # difference in classification result of rnn and nn
    #rnn_class = [np.where(rnn_predictions[i] == np.max(rnn_predictions[i])) for i in range(sample_num)]
    #nn_class = [np.where(nn_predictions[i] == np.max(nn_predictions[i])) for i in range(sample_num)]

    # Entropy calculation
    entropy_calculation = (lambda x: - x * math.log(x) if x>0 else 0)
    rnn_entropy = [[i,sum(map(entropy_calculation, rnn_predictions[i]))] for i in range(sample_num)]
    rnn_entropy.sort(key=lambda x: x[1], reverse=True)
    nn_entropy = [[i,sum(map(entropy_calculation, nn_predictions[i]))] for i in range(sample_num)]
    nn_entropy.sort(key=lambda x: x[1], reverse=True)

    # difference calculation
    rnn_prediction_sorted = [sorted(rnn_predictions[i], reverse=True) for i in range(sample_num)]
    rnn_difference = [[i,abs(rnn_prediction_sorted[i][0] - rnn_prediction_sorted[i][1])] for i in range(sample_num)]
    rnn_difference.sort(key=lambda x: x[1])

    nn_prediction_sorted = [sorted(nn_predictions[i][:], reverse=True) for i in range(sample_num)]
    nn_difference = [[i,abs(nn_prediction_sorted[i][0] - nn_prediction_sorted[i][1])] for i in range(sample_num)]
    nn_difference.sort(key=lambda x: x[1])


    rnn_entropy_set = np.array(rnn_entropy)[:15, 0]
    nn_entropy_set = np.array(nn_entropy)[:15, 0]

    rnn_difference_set = np.array(rnn_difference)[:15, 0]
    nn_difference_set = np.array(nn_difference)[:15, 0]

    sample_set = list(set(rnn_entropy_set.tolist() + nn_entropy_set.tolist() + rnn_difference_set.tolist() + nn_difference_set.tolist()))
    number = np.array(sample_set)
    result = sorted(np.array(rnn_prediction_sorted)[:,0])
    if(result[30]>0.9 or rnn_difference[30][1]>0.85):
        stop_flag = 1
    else:
        stop_flag = 0

    return number, stop_flag

def adjust_data_in_iteration_st(train_index, test_index, sample_selected, index):
    sample_num = len(sample_selected)
    if train_index.shape[0]>0:
        train_index_next = np.append(train_index, test_index[sample_selected.tolist()], axis = 0)
    else:
        train_index_next = test_index[sample_selected.tolist()]
    mask = np.ones(index.shape, dtype=bool)
    mask[train_index_next] = False
    test_index_next = index[mask]
    return train_index_next, test_index_next


def load_training_data(data_path):

    with open(data_path +'train_data.pkl', 'rb') as training_file:
        training_signals, training_time, training_label,training_position= pickle.load(training_file)
    training_signals = np.array(training_signals.astype('float32'))
    training_time = np.array(training_time.astype('float32'))
    training_label = np.array(training_label.astype('float32'))
    return training_signals, training_time, training_label, training_position

def load_testing_data(data_path):
    with  open(data_path +'test_data.pkl', 'rb') as test_file:
        test_signals, test_time, test_label, test_position = pickle.load(test_file)
    test_signals = np.array(test_signals.astype('float32'))
    test_time = np.array(test_time.astype('float32'))
    test_label = np.array(test_label.astype('float32'))
    return test_signals, test_time, test_label, test_position

def rnn_main():
    train_data_path = './train_data/'
    test_data_path = './test_data/'

    train_feature = []

    import numpy as np

    def _load_data(data, n_prev=100):
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

    def train_test_split(df, test_size=0.1):
        """
        This just splits data to training and testing parts
        """
        ntrn = round(len(df) * (1 - test_size))

        X_train, y_train = _load_data(df.iloc[0:ntrn])
        X_test, y_test = _load_data(df.iloc[ntrn:])

        return (X_train, y_train), (X_test, y_test)

        (X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data

        # and now train the model
        # batch_size should be appropriate to your memory size
        # number of epochs should be higher for real world problems
        model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)

        predicted = model.predict(X_test)
        rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

        # and maybe plot it
        pd.DataFrame(predicted[:100]).plot()
        pd.DataFrame(y_test[:100]).plot()







if __name__ =='__main__':
#    test_cross_dataset()
    active_learning_st()
    test_all_dataset_baised()
    test_all_dataset()




