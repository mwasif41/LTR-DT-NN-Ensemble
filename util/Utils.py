from collections import Counter
import numpy as np


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
