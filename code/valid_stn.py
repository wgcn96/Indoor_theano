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
from net_vggWithSTN import *
from function import *



net_prefix = 'final/'

# for debug
wangchendebug = 1
prefix = 'My debug output {}: '

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/Indoor/models/stn_best_model_data_aug.pkl'
copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1


net_final_out,net_global_out,net_inc1_out,net_inc2_out = buildSpatialTransformerNet()
net_vgg_last_layer = net_final_out

lasagneLayerList = lasagne.layers.get_all_layers(net_vgg_last_layer)
lasagne.layers.set_all_param_values(net_vgg_last_layer,copyParamsList,trainable=True)


print prefix.format(wangchendebug) + 'build model and copy params finish!'
wangchendebug += 1


X = T.tensor4()
y = T.ivector()

valueLayerList = [lasagneLayerList[42],lasagneLayerList[43],lasagneLayerList[66],lasagneLayerList[67],lasagneLayerList[92]]
theta_1,theta_2,transform1,transform2,y_hat = lasagne.layers.get_output(valueLayerList,X,deterministic=True)
eval = theano.function([X], [theta_1,theta_2,transform1,transform2,y_hat], on_unused_input='warn')

def eval_input(data,data_y,N,n=10):
    count = 0.
    shuffle = np.random.permutation(N)     # 用random函数，打乱数据
    step = int(np.ceil(N/float(n)))
    for i in range(step):
        idx = range(i * n, np.minimum((i + 1) * n, N))
        pos = shuffle[idx]
        currentImage = data[pos]
        y = data_y[pos]
        theta_1, transform1, theta_2, transform2, y_hat = eval(currentImage)
        preds = np.argmax(y_hat, axis=-1)
        count += np.sum(y == preds)

    accu = count/N
    result =  'total estimate {0} images, the truth times is {1}, accu is: {2}'.format(N,count,accu)
    print result




eval_input(data['X_test'],data['y_test'],test_num,12)
eval_input(data['X_train'],data['y_train'],train_num,12)

#print theta w b
