# -*- coding: UTF-8 -*-

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import softmax, linear

import lasagne
import numpy as np

net_input = InputLayer((None, 3, 224, 224))

def build_vgg_model(prefix,inputLayer=net_input,classificationFlag=False,stnFlag=False,dropout_ratio = 0.5):
    net = {}

    net[prefix + 'input'] = inputLayer

    net[prefix + 'conv1_1'] = ConvLayer(
        net[prefix + 'input'], 64, 3, pad=1, flip_filters=False, name=prefix + 'conv1_1')

    net[prefix + 'conv1_2'] = ConvLayer(
        net[prefix + 'conv1_1'], 64, 3, pad=1, flip_filters=False, name=prefix + 'conv1_2')

    net[prefix + 'pool1'] = PoolLayer(net[prefix + 'conv1_2'], 2, name=prefix + 'pool1')

    net[prefix + 'conv2_1'] = ConvLayer(
        net[prefix + 'pool1'], 128, 3, pad=1, flip_filters=False, name=prefix + 'conv2_1')

    net[prefix + 'conv2_2'] = ConvLayer(
        net[prefix + 'conv2_1'], 128, 3, pad=1, flip_filters=False, name=prefix + 'conv2_2')

    net[prefix + 'pool2'] = PoolLayer(net[prefix + 'conv2_2'], 2, name=prefix + 'pool2')

    net[prefix + 'conv3_1'] = ConvLayer(
        net[prefix + 'pool2'], 256, 3, pad=1, flip_filters=False ,name=prefix + 'conv3_1')

    net[prefix + 'conv3_2'] = ConvLayer(
        net[prefix + 'conv3_1'], 256, 3, pad=1, flip_filters=False, name=prefix + 'conv3_2')

    net[prefix + 'conv3_3'] = ConvLayer(
        net[prefix + 'conv3_2'], 256, 3, pad=1, flip_filters=False , name=prefix + 'conv3_3')

    net[prefix + 'pool3'] = PoolLayer(net[prefix +'conv3_3'], 2, name=prefix + 'pool3')

    net[prefix +'conv4_1'] = ConvLayer(
        net[prefix +'pool3'], 512, 3, pad=1, flip_filters=False ,name=prefix +'conv4_1')

    net[prefix +'conv4_2'] = ConvLayer(
        net[prefix +'conv4_1'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv4_2')

    net[prefix +'conv4_3'] = ConvLayer(
        net[prefix +'conv4_2'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv4_3')

    net[prefix +'pool4'] = PoolLayer(net[prefix +'conv4_3'], 2 ,name=prefix +'pool4')

    net[prefix +'conv5_1'] = ConvLayer(
        net[prefix +'pool4'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv5_1')

    net[prefix +'conv5_2'] = ConvLayer(
        net[prefix +'conv5_1'], 512, 3, pad=1, flip_filters=False, name=prefix +'conv5_2')

    net[prefix +'conv5_3'] = ConvLayer(
        net[prefix +'conv5_2'], 512, 3, pad=1, flip_filters=False ,name=prefix +'conv5_3')

    net[prefix +'pool5'] = PoolLayer(net[prefix +'conv5_3'], 2,name=prefix +'pool5')

    if stnFlag == False:
        net[prefix +'fc6'] = DenseLayer(net[prefix +'pool5'], num_units=4096, name=prefix +'fc6')

        net[prefix +'fc6_dropout'] = DropoutLayer(net[prefix +'fc6'], p=dropout_ratio, name=prefix +'fc6_dropout')

        net[prefix +'fc7'] = DenseLayer(net[prefix +'fc6_dropout'], num_units=4096, name=prefix +'fc7')

        net[prefix +'fc7_dropout'] = DropoutLayer(net[prefix +'fc7'], p=dropout_ratio, name=prefix +'fc7_dropout')

        if classificationFlag == True:
            net[prefix +'fc8'] = DenseLayer(
                net[prefix +'fc7_dropout'], num_units=67, nonlinearity=None, name=prefix +'fc8')

            net[prefix +'prob'] = NonlinearityLayer(net[prefix +'fc8'], softmax, name=prefix +'prob')

    return net


def buildSTN(stn_downsample_factor=1):

    st_prefix = 'st/'

    b1 = [0.5 , 0, 0,  0, 0.5, 0]
    b2 = [0.25 , 0, 0 , 0, 0.25, 0]
    b1 = np.asarray(b1,dtype='float32')
    b2 = np.asarray(b2,dtype='float32')

    googleNetDict_st = build_vgg_model(inputLayer=net_input, prefix=st_prefix,stnFlag=True, classificationFlag=False,dropout_ratio=0.5)

    net_st_pre = googleNetDict_st['st/pool5']

    #net_st_final_conv = ConvLayer(net_st_pre,num_filters=128,filter_size=1,name=st_prefix+'final_conv')

    net_st_pre_theta_fc128 = DenseLayer(net_st_pre,128,name=st_prefix+'final_fc')

    net_st_theta_1_fc = DenseLayer(net_st_pre_theta_fc128,6,name=st_prefix+'theta_1',b=b1, W=lasagne.init.Constant(0.0),nonlinearity=None)

    net_st_theta_2_fc = DenseLayer(net_st_pre_theta_fc128,6,name=st_prefix+'theta_2',b=b2, W=lasagne.init.Constant(0.0),nonlinearity=None)

    net_stn_1 = lasagne.layers.TransformerLayer(net_input, net_st_theta_1_fc, name=st_prefix+'stn_1',downsample_factor=stn_downsample_factor)

    net_stn_2 = lasagne.layers.TransformerLayer(net_input, net_st_theta_2_fc, name=st_prefix+'stn_2',downsample_factor=stn_downsample_factor)

    #print "Transformer network output shape: ", net_stn_1.output_shape
    return net_stn_1,net_stn_2

def buildSpatialTransformerNet():

    #st_prefix = 'st/'
    inc1_prefix = 'inc1/'
    inc2_prefix = 'inc2/'
    global_prefix = 'global/'

    net_stn_1, net_stn_2 = buildSTN()

    net_inc1_Dict = build_vgg_model(inputLayer=net_stn_1,prefix=inc1_prefix,dropout_ratio=0.75,stnFlag=False,classificationFlag=True)
    net_inc2_Dict = build_vgg_model(inputLayer=net_stn_2,prefix=inc2_prefix,dropout_ratio=0.75,stnFlag=False,classificationFlag=True)
    net_global_Dict = build_vgg_model(inputLayer=net_input, prefix=global_prefix, dropout_ratio=0.75, stnFlag=False,classificationFlag=True)

    net_inc1_output = net_inc1_Dict[inc1_prefix+'fc7_dropout']
    net_inc2_output = net_inc2_Dict[inc2_prefix+'fc7_dropout']
    net_global_output = net_global_Dict[global_prefix+'fc7_dropout']

    #net_final_concat = ElemwiseSumLayer([net_global_output,net_inc1_output,net_inc2_output], [0.6,0.2,0.2],name='final/concat/output')

    net_final_concat = ConcatLayer([net_global_output,net_inc1_output,net_inc2_output],name='final/concat/output')

    net_final_fc = DenseLayer(net_final_concat,num_units=67,nonlinearity=None,
                                         name='final/fc')

    net_final_prob = NonlinearityLayer(net_final_fc,
                                    nonlinearity=softmax, name='final/prob')

    net_global_prob = net_global_Dict[global_prefix+'prob']
    net_inc1_prob = net_inc1_Dict[inc1_prefix+'prob']
    net_inc2_prob = net_inc2_Dict[inc2_prefix+'prob']

    return net_final_prob,net_global_prob,net_inc1_prob,net_inc2_prob

if __name__ == '__main__':
    final_out,global_out,inc1_out,inc2_out = buildSpatialTransformerNet()
    lasagneLayers = lasagne.layers.get_all_layers(final_out)
    lasagneParams = lasagne.layers.get_all_params(final_out)
    print 'check finish!'