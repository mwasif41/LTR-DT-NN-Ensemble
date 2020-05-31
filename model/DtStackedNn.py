from model.NeuralNetwork import DeepNeuralNetwork
from constant.Constant import INPUT_DIMS_MQ2008
from model.DecisionTree import DecisionTree


class DtStackedNn:
    nn_model = []
    dt_model = []

    def __init__(self):
        self.nn_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008 + 1)
        self.dt_model = DecisionTree(200)

    def fit(self, train_data):
        self.dt_model.fit(train_data)
        pred = self.dt_model.predict(train_data)
        stacked_data = train_data
        stacked_data[len(stacked_data.columns)] = pred
        self.nn_model.fit(stacked_data)

    def predict(self, test_data):
        pred = self.dt_model.predict(test_data)
        stacked_data = test_data
        stacked_data[len(stacked_data.columns)] = pred
        return self.nn_model.predict(stacked_data)
