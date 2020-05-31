from model.NeuralNetwork import DeepNeuralNetwork
import numpy as np
from constant.Constant import INPUT_DIMS_MQ2008
from model.DecisionTree import DecisionTree


class NnDtBagger:
    nn_model = []
    dt_model = []
    nn_weight = 0.5

    def __init__(self, nn_weight):
        self.nn_weight = nn_weight
        self.nn_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008)
        self.dt_model = DecisionTree(100)

    def fit(self, train_data):
        # TODO:  do some data processing if needed (data division)
        self.nn_model.fit(train_data)
        self.dt_model.fit(train_data)

    def predict(self, test_data):
        pred = self.nn_weight * np.asarray(self.nn_model.predict(test_data)) + (1 - self.nn_weight) * np.asarray(
            self.dt_model.predict(test_data))
        return pred.astype(int)
