from model.NeuralNetwork import DeepNeuralNetwork
from constant.Constant import INPUT_DIMS_MQ2008
from sklearn.ensemble import AdaBoostClassifier
from util.Utils import get_data_params
import numpy as np


class NnBoostedDt:
    nn_model = []
    dt_model = []

    def __init__(self):
        self.nn_model = DeepNeuralNetwork(INPUT_DIMS_MQ2008)
        self.dt_model = AdaBoostClassifier(n_estimators=300, random_state=0)

    def fit(self, train_data):
        self.nn_model.fit(train_data)
        pred = self.nn_model.predict(train_data)
        train_x, train_y, train_q = get_data_params(train_data)
        boosted_data = train_data
        boosted_data[0] = pred
        self.dt_model.fit(boosted_data, np.subtract(pred, train_y))

    def predict(self, test_data):
        pred_nn = self.nn_model.predict(test_data)
        pred_dt = self.dt_model.predict(test_data)
        return pred_nn + pred_dt
