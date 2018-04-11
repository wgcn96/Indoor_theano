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
from googleNet import *
from function import *


np.random.seed(123)

prefix = 'googleNet/'
prototxt = '/media/wangchen/newdata1/wangchen/work/Indoor/caffemodel/MyGoogleNet/deploy_googlenet_places365.prototxt'
weight = '/media/wangchen/newdata1/wangchen/work/Indoor/caffemodel/places365-master/googlenet_places365.caffemodel'
saveURL = '/media/wangchen/newdata1/wangchen/work/Indoor/models/caffe_googleNet_places.pkl'
caffeParamList = getOriParams(prototxt, weight)

net_google_Dict = build_model(input=net_input, prefix=prefix, lastClassNum=67, classificationFlag=True)
net_google_last_layer = net_google_Dict[prefix+'prob']

lasagneLayerList = lasagne.layers.get_all_layers(net_google_last_layer)
lasagneParamList = lasagne.layers.get_all_params(net_google_last_layer, trainable=True)
lasagneParamvaluesList = lasagne.layers.get_all_param_values(net_google_last_layer, trainable=True)

checkDimes(lasagneParamvaluesList,caffeParamList)

caffeFCTransform(caffeParamList,[114])

checkDimes(lasagneParamvaluesList,caffeParamList)

lasagne.layers.set_all_param_values(net_google_last_layer, caffeParamList)

targetAllParams = lasagne.layers.get_all_param_values(net_google_last_layer, trainable = True)

copyBool, copyNum = checkParams(targetAllParams,caffeParamList)

saveModel(saveURL,targetAllParams)

print 'load model finish!',copyBool,copyNum