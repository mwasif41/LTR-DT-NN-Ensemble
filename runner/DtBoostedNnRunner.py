from sklearn.model_selection import train_test_split
from util.Utils import read_dataset_as_df
from constant.Constant import DATASET_MQ2008_PATH
from constant.Constant import MQ2008_TSV_FILE_NAME
from util.Utils import get_data_params
from util.Utils import calculate_ndcg
from util.Utils import calculate_map
from model.DtBoostedNn import DtBoostedNn

''''
Main Running logic for the Deep NN model and Decision tree with ensemble technique

Technique : Boosted Stacking (DT -> NN)
Dataset : MQ2008

'''

print(":: Boosted Stacking (DT -> NN) started ::")
df = read_dataset_as_df(DATASET_MQ2008_PATH + MQ2008_TSV_FILE_NAME)
# Dividing the data
train, test = train_test_split(df, test_size=0.7)
test_x ,test_y , test_q = get_data_params(test)
model = DtBoostedNn()
model.fit(train)
pred = model.predict(test)
print("predicted value :" , pred)
ndcg = calculate_ndcg(pred, test_y)
mAP = calculate_map(pred, test_y)

print('NDCG For Deep NN and DT Boosted Stacking (DT -> NN) :', ndcg)
print('MAP For Deep NN and DT Boosted Stacking (DT -> NN) :', mAP)