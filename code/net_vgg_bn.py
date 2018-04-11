# -*- coding: UTF-8 -*-

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear

import lasagne
import numpy as np

net_input = InputLayer((None, 3, 224, 224))

def build_vgg_model(prefix,inputLayer=net_input,classificationFlag=False,stnFlag=False,dropout_ratio = 0.5):
    net = {}

    net[prefix + 'input'] = inputLayer

    net[prefix + 'conv1_1'] = batch_norm( ConvLayer(
        net[prefix + 'input'], 64, 3, pad=1, flip_filters=False, name=prefix + 'conv1_1') ,name=prefix + 'conv1_1_bn')

    net[prefix + 'conv1_2'] = batch_norm( ConvLayer(
        net[prefix + 'conv1_1'], 64, 3, pad=1, flip_filters=False, name=prefix + 'conv1_2') ,name=prefix + 'conv1_2_bn')

    net[prefix + 'pool1'] = PoolLayer(net[prefix + 'conv1_2'], 2, name=prefix + 'pool1')

    net[prefix + 'conv2_1'] = batch_norm( ConvLayer(
        net[prefix + 'pool1'], 128, 3, pad=1, flip_filters=False, name=prefix + 'conv2_1') ,name=prefix + 'conv2_1_bn')

    net[prefix + 'conv2_2'] = batch_norm( ConvLayer(
        net[prefix + 'conv2_1'], 128, 3, pad=1, flip_filters=False, name=prefix + 'conv2_2') , name=prefix + 'conv2_2_bn')

    net[prefix + 'pool2'] = PoolLayer(net[prefix + 'conv2_2'], 2, name=prefix + 'pool2')

    net[prefix + 'conv3_1'] = batch_norm( ConvLayer(
        net[prefix + 'pool2'], 256, 3, pad=1, flip_filters=False ,name=prefix + 'conv3_1') ,name=prefix + 'conv3_1_bn')

    net[prefix + 'conv3_2'] = batch_norm( ConvLayer(
        net[prefix + 'conv3_1'], 256, 3, pad=1, flip_filters=False, name=prefix + 'conv3_2') , name=prefix + 'conv3_2_bn')

    net[prefix + 'conv3_3'] = batch_norm( ConvLayer(
        net[prefix + 'conv3_2'], 256, 3, pad=1, flip_filters=False , name=prefix + 'conv3_3') , name=prefix + 'conv3_3_bn')

    net[prefix + 'pool3'] = PoolLayer(net[prefix +'conv3_3'], 2, name=prefix + 'pool3')

    net[prefix +'conv4_1'] = batch_norm( ConvLayer(
        net[prefix +'pool3'], 512, 3, pad=1, flip_filters=False ,name=prefix +'conv4_1') ,name=prefix +'conv4_1_bn')

    net[prefix +'conv4_2'] = batch_norm( ConvLayer(
        net[prefix +'conv4_1'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv4_2') , name=prefix +'conv4_2_bn')

    net[prefix +'conv4_3'] = batch_norm( ConvLayer(
        net[prefix +'conv4_2'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv4_3') , name=prefix +'conv4_3_bn')

    net[prefix +'pool4'] = PoolLayer(net[prefix +'conv4_3'], 2 ,name=prefix +'pool4')

    net[prefix +'conv5_1'] = batch_norm( ConvLayer(
        net[prefix +'pool4'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv5_1') , name=prefix +'conv5_1_bn')

    net[prefix +'conv5_2'] = batch_norm( ConvLayer(
        net[prefix +'conv5_1'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv5_2') , name=prefix +'conv5_2_bn')

    net[prefix +'conv5_3'] = batch_norm( ConvLayer(
        net[prefix +'conv5_2'], 512, 3, pad=1, flip_filters=False ,name=prefix +'conv5_3') ,name=prefix +'conv5_3_bn')

    net[prefix +'pool5'] = PoolLayer(net[prefix +'conv5_3'], 2,name=prefix +'pool5')

    if stnFlag == False:
        net[prefix +'fc6'] = batch_norm( DenseLayer(net[prefix +'pool5'], num_units=4096, name=prefix +'fc6') , name=prefix +'fc6_bn')

        net[prefix +'fc6_dropout'] = DropoutLayer(net[prefix +'fc6'], p=dropout_ratio, name=prefix +'fc6_dropout')

        net[prefix +'fc7'] = batch_norm( DenseLayer(net[prefix +'fc6_dropout'], num_units=4096, name=prefix +'fc7') , name=prefix +'fc7_bn')

        net[prefix +'fc7_dropout'] = DropoutLayer(net[prefix +'fc7'], p=dropout_ratio, name=prefix +'fc7_dropout')

        if classificationFlag == True:
            net[prefix +'fc8'] = DenseLayer(
                net[prefix +'fc7_dropout'], num_units=67, nonlinearity=None, name=prefix +'fc8')

            net[prefix +'prob'] = NonlinearityLayer(net[prefix +'fc8'], softmax, name=prefix +'prob')

    return net
