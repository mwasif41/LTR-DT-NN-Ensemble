#!/usr/bin/env python
# coding: utf-8

# importing all the necessary modules
import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_svmlight_file
from constant import Constant
from util.Utils import print_dataset_statistics




# Load datasets in the svmlight / libsvm format into sparse CSR matrix
# This format is a text-based format, with one sample per line.
# It does not store zero valued features hence is suitable for sparse dataset.
# The first element of each line can be used to store a target variable to predict.

def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return X.todense(), y, queries


# function to create csv file from the processed data

def dump_to_file(out_file_name, X, y, queries):
    all = np.hstack((y.reshape(-1, 1), queries.reshape(-1, 1), X))
    pd.DataFrame(all).sort_values(by=[1]).to_csv(out_file_name, sep='\t', header=False, index=False)


def mq2008(src_path, dst_path):
    """
    0 - label, 1 - qid, ...features...
    ----------------------------------
    Characteristics of dataset mq2008 train
    rows x columns (9630, 46)
    sparsity: 47.2267370987
    y distribution
    Counter({0.0: 7820, 1.0: 1223, 2.0: 587})
    num samples in queries: minimum, median, maximum
    (5, '8.0', 121)
    ----------------------------------
    ----------------------------------
    Characteristics of dataset mq2008 test
    rows x columns (2874, 46)
    sparsity: 46.1128256331
    y distribution
    Counter({0.0: 2319, 1.0: 378, 2.0: 177})
    num samples in queries: minimum, median, maximum
    (6, '14.5', 119)
    ----------------------------------
    """
    train_file = os.path.join(src_path, "train.txt")
    test_file = os.path.join(src_path, "test.txt")

    train_out_file = os.path.join(dst_path, "train.tsv")
    test_out_file = os.path.join(dst_path, "test.tsv")

    X, y, queries = process_libsvm_file(train_file)
    print_dataset_statistics(X, y, queries, "mq2008 train")
    dump_to_file(train_out_file, X, y, queries)

    X, y, queries = process_libsvm_file(test_file)
    print_dataset_statistics(X, y, queries, "mq2008 test")
    dump_to_file(test_out_file, X, y, queries)


# Uncomment these for the TREC2004 data

# src_path = 'Dataset/TREC2004/All/'
# dst_path = 'Dataset/TREC2004/Processed/All/'
#
# data_file_path = os.path.join(src_path, "TD2004.txt")
# data_out_path = os.path.join(dst_path, "TD2004.tsv")
#
# X, y, queries = process_libsvm_file(data_file_path)
# print_dataset_statistics(X, y, queries, "mq2004 all")

# dump_to_file(data_out_path, X, y, queries)

src_path = Constant.DATASET_MQ2008_BASE_PATH
dst_path = Constant.DATASET_MQ2008_PATH

data_file_path = os.path.join(src_path, "min.txt")
data_out_path = os.path.join(dst_path, Constant.MQ2008_TSV_FILE_NAME)

X, y, queries = process_libsvm_file(data_file_path)
print_dataset_statistics(X, y, queries, "mq2008 all")

dump_to_file(data_out_path, X, y, queries)
