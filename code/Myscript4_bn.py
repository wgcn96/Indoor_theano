# -*- coding: UTF-8 -*-

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm,BatchNormLayer
from lasagne.nonlinearities import softmax

l1_in = InputLayer((64, 768),name='l1_in')
l1_dense = DenseLayer(l1_in, num_units=500,name='l1_dense')

l2_in = InputLayer((64, 768),name='l2_in')
l2_batch = batch_norm(DenseLayer(l2_in, num_units=500,name='l2_dense'),name='l2_batch')

l1_layerList = lasagne.layers.get_all_layers(l1_dense)
l2_layerList = lasagne.layers.get_all_layers(l2_batch)

batch_norm_Layer = l2_layerList[2]
batch_norm_params = batch_norm_Layer.params



l1_layerParams = lasagne.layers.get_all_params(l1_dense)
l2_layerParams = lasagne.layers.get_all_params(l2_batch)
l2_trainableParams = lasagne.layers.get_all_param_values(l2_batch,trainable=True)


print 'finish!'