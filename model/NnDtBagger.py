from model.NeuralNetwork import DeepNeuralNetwork
import numpy as np


class NnDtBagger:
    nn_model = []
    dt_model = []
    nn_weight = 0.5

    def __init__(self, nn_weight):
        self.nn_weight = nn_weight
        self.nn_model = DeepNeuralNetwork()
        # TODO change this to DT implementation
        self.dt_model = DeepNeuralNetwork()

    def fit(self, train_data):
        # TODO:  do some data processing if needed (data division)
        self.nn_model.fit(train_data)
        self.dt_model.fit(train_data)

    def predict(self, test_data):
        return self.nn_weight * np.asarray(self.nn_model.predict(test_data)) + (1 - self.nn_weight) * np.asarray(
            self.dt_model.predict(test_data))
