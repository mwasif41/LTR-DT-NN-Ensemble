from catboost import CatBoostClassifier, Pool
from util.Utils import get_data_params
from util.Utils import normalize_data


class DecisionTree:
    train_pool = []
    model = []

    def __init__(self, iterations):
        self.model = CatBoostClassifier(iterations=iterations,
                           learning_rate=0.5,
                           loss_function='MultiClass')

    def fit(self, train_data):
        train_x, train_y, train_q = get_data_params(train_data)
        self.train_pool = Pool(data=normalize_data(train_x), label=train_y.astype(int))
        self.model.fit(X=self.train_pool)

    def predict(self, test_data):
        test_x, test_y, test_q = get_data_params(test_data)
        return self.model.predict(Pool(data=normalize_data(test_x))).ravel()
