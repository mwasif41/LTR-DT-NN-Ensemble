from sklearn.model_selection import train_test_split
from util.Utils import read_dataset_as_df
from constant.Constant import DATASET_MQ2008_PATH
from constant.Constant import MQ2008_TSV_FILE_NAME
from model.NnStackedDt import NnStackedDt
from util.Utils import get_data_params
from util.Utils import calculate_ndcg
from util.Utils import calculate_map


''''
Main Running logic for the Deep NN model and Decision tree with ensemble technique

Technique : Stacking (NN -> DT)
Dataset : MQ2008

'''

print(":: Stacking (NN -> DT) started ::")
df = read_dataset_as_df(DATASET_MQ2008_PATH + MQ2008_TSV_FILE_NAME)
# Dividing the data
train, test = train_test_split(df, test_size=0.7)
train_x, train_y, train_q = get_data_params(train)
test_x ,test_y , test_q = get_data_params(test)
model = NnStackedDt()
model.fit(train)
pred = model.predict(test)
ndcg = calculate_ndcg(pred, test_y)
mAP = calculate_map(pred, test_y)

print('NDCG For Deep NN and DT Stacking (NN -> DT) :', ndcg)
print('MAP For Deep NN and DT Stacking (NN -> DT) :', mAP)