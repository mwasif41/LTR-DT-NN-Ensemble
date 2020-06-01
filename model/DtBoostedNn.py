from model.NeuralNetwork import DeepNeuralNetwork
from constant.Constant import INPUT_DIMS_MQ2008
from model.DecisionTree import DecisionTree
from sklearn.ensemble import AdaBoostClassifier
from util.Utils import get_data_params
import numpy as np


class DtBoostedNn:
    nn_model = []
    dt_model = []

    def __init__(self):
        self.dt_model = AdaBoostClassifier(n_estimators=300, random_state=0)
        self.nn_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008)

    def fit(self, train_data):
        train_x, train_y, train_q = get_data_params(train_data)
        self.dt_model.fit(train_x, train_y)
        pred = self.dt_model.predict(train_x)
        boosted_data = train_data
        boosted_data[0] = pred
        self.nn_model.fit_with_labels(train_x, np.subtract(pred, train_y))

    def predict(self, test_data):
        test_x, test_y, test_q = get_data_params(test_data)
        pred_nn = self.nn_model.predict(test_x)
        pred_dt = self.dt_model.predict(test_x)
        return pred_nn + pred_dt
