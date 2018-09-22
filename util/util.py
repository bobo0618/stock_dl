import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Assume use csv from yahoo
def read_data(data_file, time_predict):
    '''
    :param data_file:
    :param time_predict: '2018-07'
    :return: train_set, test_set
    '''

    dataset = pd.read_csv(data_file, index_col='Date',parse_dates=['Date'])
    print(dataset.head())
    train_set = dataset[:time_predict].iloc[:,1:2].values
    test_set = dataset[time_predict:].iloc[:,1:2].values
    return train_set, test_set

def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaler, scaled_data

def get_training_testing_data(data_file, time_predict, window_size):
    train_set, test_set = read_data(data_file, time_predict)
    print ("train_set {}".format(train_set.shape))
    sc, scaled_training_data = scale_data(train_set)
    x_train = []
    y_train = []
    for i in range(window_size, train_set.shape[0]):
        x_train.append(scaled_training_data[i - window_size:i, 0])
        y_train.append(scaled_training_data[i,0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train
