from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import math
import numpy as np
from keras.models import model_from_json

class LstmModel:
    def __init__(self, model_json_path, model_weight_path, seqence_len):
        self.model_json_path = model_json_path
        self.model_weight_path = model_weight_path
        self.sequence_len = seqence_len
        self.regressor = Sequential()

    def get_model(self):
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.sequence_len, 1)))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(1))
        regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
        return regressor

    def save_model(self):
        model_json = self.regressor.to_json()
        with open(self.model_json_path, 'w') as json_file:
            json_file.write(model_json)

        self.regressor.save_weights(self.model_weight_path)

    def load_model(self):
        json_file = open(self.model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.regressor = model_from_json(loaded_model_json)
        self.regressor.load_weights(self.model_weight_path)
