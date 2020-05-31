from model.NeuralNetwork import DeepNeuralNetwork
import numpy as np
from constant.Constant import INPUT_DIMS_MQ2008


class DtStackedNn:
    nn_model = []
    dt_model = []

    def __init__(self):
        self.nn_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008 + 1)
        # TODO change this to DT implementation
        self.dt_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008)

    def fit(self, train_data):
        # TODO:  This will be done once got decision tree classifier
        self.dt_model.fit(train_data)
        self.nn_model.fit(train_data)

    def predict(self, test_data):
        # TODO:  This will be done once got decision tree classifier
        return  np.asarray(self.dt_model.predict(test_data))
