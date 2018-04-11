# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda1"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np


from collections import OrderedDict



import sys
script_path = '/media/wangchen/newdata1/wangchen/work/Indoor/code'
sys.path.append(script_path)
from loadData import *
from vggWithSTN import *
from function import *
from dataBase import *


net_prefix = 'final/'

# for debug
wangchendebug = 1
prefix = 'My debug output {}: '

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/copy/models/79.6%/stn_best_model_2.pkl'
copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1

db,cursor = connectdb()

net_final_out,net_global_out,net_inc1_out,net_inc2_out = buildSpatialTransformerNet()
net_vgg_last_layer = net_final_out

lasagneLayerList = lasagne.layers.get_all_layers(net_vgg_last_layer)
lasagne.layers.set_all_param_values(net_vgg_last_layer,copyParamsList,trainable=True)


print prefix.format(wangchendebug) + 'build model and copy params finish!'
wangchendebug += 1


X = T.tensor4()
y = T.ivector()

concatFeature = lasagne.layers.get_output(lasagneLayerList[90],X,deterministic=True)
eval = theano.function([X], [concatFeature], on_unused_input='warn')

def extract_feature(data_X, N, tablename, parallelNum=10):
    feature = np.zeros((N,12288))
    num_batches = int(np.ceil(N / float(parallelNum)))
    for i in range(num_batches):
        idx = range(i * parallelNum, np.minimum((i + 1) * parallelNum, N))
        data_batch = data_X[idx]
        feature_batch = eval(data_batch)
        feature[idx] = feature_batch[:]

    # 将feature写入数据库
    wirte_feature_to_db(db=db,cursor=cursor,table_name=tablename,featurename='FEATURE1',feature=feature)

    return feature


'''
train_feature_file = '/media/wangchen/newdata1/wangchen/work/Indoor/features/stn_train_feature.pkl'
test_feature_file = '/media/wangchen/newdata1/wangchen/work/Indoor/features/stn_test_feature.pkl'

train_feature = reloadModel(train_feature_file)
test_feature = reloadModel(test_feature_file)
'''
test_feature = extract_feature(data['X_test'], 1340, testtable, 10)
train_feature = extract_feature(data['X_train'], 5360, traintable, 10)

test_feature_db = read_feature(db,cursor,testtable,'FEATURE1',1340)
train_feature_db = read_feature(db,cursor,traintable,'FEATURE1',5360)
result1 = (test_feature == test_feature_db).all()
result2 = (train_feature == train_feature_db).all()
#train_feature_pca,test_feature_pca = myPCA(train_feature, test_feature, 0.99)
#m_precision = mySVM(train_feature_pca,test_feature_pca,data['y_train'],data['y_test'])
#saveModel(test_feature_file,test_feature)
#saveModel(train_feature_file,train_feature)

closedb(db)
print result1,result2,"finish !"
