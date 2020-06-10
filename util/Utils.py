from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# Checking sparsity of the feature / dataset keeping in mind that data can be sparsed

def sparsity(X):
    number_of_nan = np.count_nonzero(np.isnan(X))
    number_of_zeros = np.count_nonzero(np.abs(X) < 1e-6)
    return (number_of_nan + number_of_zeros) / float(X.shape[0] * X.shape[1]) * 100.


# A generic method to check different aspects of any dataset

def print_dataset_statistics(X, y, queries, name):
    print('----------------------------------')
    print("Characteristics of dataset " + name)
    print("rows x columns " + str(X.shape))
    print("sparsity: " + str(sparsity(X)))
    print("y distribution")
    print(Counter(y))
    print('----------------------------------')


def read_dataset(file_name):
    df = read_dataset_as_df(file_name)
    return get_data_params(df)


def get_data_params(df):
    y = df[0].values
    queries = df[1].values
    X = df.iloc[:, 1:].values
    return X, y, queries


def read_dataset_as_df(file_name):
    return pd.read_csv(file_name, sep='\t', header=None)


def normalize_data(data):
    sc = StandardScaler()
    return sc.fit_transform(data)


def encode_label(label):
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder()
    return ohe.fit_transform(label.reshape(-1, 1)).toarray()


def decode_label(encoded_label):
    labels = list()
    for i in range(len(encoded_label)):
        labels.append(np.argmax(encoded_label[i]))
    return labels


def calculate_accuracy(pred, test):
    return accuracy_score(pred, test)


def calculate_ndcg(pred, test):
    return ndcg_score(np.asarray([test]), np.asarray([pred]))


def calculate_map(pred, test):
    precision = precision_score(pred, test, average='micro')
    recall = recall_score(pred, test, average='micro', zero_division=1)
    return precision * recall
