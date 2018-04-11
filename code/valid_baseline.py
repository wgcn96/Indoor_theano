# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda0"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np
import datetime
from collections import OrderedDict


import sys
script_path = '/media/wangchen/newdata1/wangchen/work/Indoor/code/'
sys.path.append(script_path)
from loadData import *
from net_vggWithSTN import *
from function import *



# for debug
wangchendebug = 1
prefix = 'My debug output {}: '
net_prefix = 'vgg/'

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/media/wangchen/newdata1/wangchen/work/Indoor/models/baseline_best_model.pkl'
copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1


net_vgg_Dict = build_vgg_model(prefix=net_prefix,inputLayer=net_input,classificationFlag=True)
net_vgg_last_layer = net_vgg_Dict[net_prefix + 'prob']

lasagneLayerList = lasagne.layers.get_all_layers(net_vgg_last_layer)
lasagne.layers.set_all_param_values(net_vgg_last_layer,copyParamsList,trainable=True)

print prefix.format(wangchendebug) + 'build model and copy params finish!'
wangchendebug += 1


X = T.tensor4()
y = T.ivector()

y_hat = lasagne.layers.get_output(net_vgg_last_layer,X,deterministic=True)
eval = theano.function([X],[y_hat])


def eval_input(data,data_y,N,n=10):
    count = 0.
    shuffle = np.random.permutation(N)     # 用random函数，打乱数据
    step = int(np.ceil(N/float(n)))
    for i in range(step):
        idx = range(i * n, np.minimum((i + 1) * n, N))
        pos = shuffle[idx]
        currentImage = data[pos]
        y = data_y[pos]
        y_hat = eval(currentImage)
        preds = np.argmax(y_hat, axis=-1)
        count += np.sum(y == preds)

    accu = count/N
    result =  'total estimate {0} images, the truth times is {1}, accu is: {2}'.format(N,count,accu)
    print result


eval_input(data['X_test'],data['y_test'],1340)
eval_input(data['X_train'],data['y_train'],5360)
