from keras.models import Sequential
from keras.layers import Dense
from util.Utils import normalize_data
from util.Utils import get_data_params
from util.Utils import encode_label
from util.Utils import decode_label

class DeepNeuralNetwork:
    model = Sequential()

    def __init__(self):
        self.model.add(Dense(40, input_dim=46, activation='relu'))
        self.model.add(Dense(30, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = ''

    def fit(self, train_data):
        train_x, train_y, train_q = get_data_params(train_data)
        train_x = normalize_data(train_x)
        train_y = encode_label(train_y)
        self.history = self.model.fit(train_x, train_y, epochs=100, batch_size=64)
        return self.history

    def predict(self, test_data):
        test_x, test_y, test_q = get_data_params(test_data)
        return decode_label(self.model.predict(test_x))
