from sklearn.model_selection import train_test_split
from util.Utils import read_dataset_as_df
from constant.Constant import DATASET_MQ2008_PATH
from constant.Constant import MQ2008_TSV_FILE_NAME
from model.NeuralNetwork import DeepNeuralNetwork
from util.Utils import get_data_params
from util.Utils import calculate_ndcg
from util.Utils import calculate_map
from constant.Constant import INPUT_DIMS_MQ2008


''''
Main Running logic for the Deep NN model with out ensemble
Dataset : MQ2008

'''

print(":: Main Running logic for the Deep NN model with out ensemble ::")
df = read_dataset_as_df(DATASET_MQ2008_PATH + MQ2008_TSV_FILE_NAME)
# Dividing the data
train, test = train_test_split(df, test_size=0.7)
train_x, train_y, train_q = get_data_params(train)
test_x ,test_y , test_q = get_data_params(test)
model = DeepNeuralNetwork(INPUT_DIMS_MQ2008)
model.fit(train)
pred = model.predict(test)
ndcg = calculate_ndcg(pred, test_y)
mAP = calculate_map(pred, test_y)

print('NDCG For Deep NN :', ndcg)
print('MAP For Deep NN :', mAP)