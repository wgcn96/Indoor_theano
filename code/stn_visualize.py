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
script_path = '/home/ubuntu/wangchen/work/Indoor/code'
sys.path.append(script_path)
from loadData import *
from net_vggWithSTN import *
from function import *

dir = '/home/ubuntu/wangchen/work/Indoor/stn_visio/'
if os.path.exists(dir)  == False:
    os.mkdir(dir)


net_prefix = 'final/'

# for debug
wangchendebug = 1
prefix = 'My debug output {}: '

data = makeDict(train_file_url,test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1

print prefix.format(wangchendebug) + 'reload params from the best model...'
wangchendebug += 1

reload_url = '/home/ubuntu/wangchen/work/models/stn_best_model.pkl'
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
    for i in range(n):
        pos = shuffle[i]
        currentImage = data[pos]
        visio_current = currentImage.reshape(224,224,3)
        currentImage = currentImage.reshape(1,3,224,224)
        y = data_y[pos]
        cv2.imwrite(dir+str(i)+'_ori_'+str(y)+'.png',visio_current)
        theta_1, transform1, theta_2, transform2, y_hat = eval(currentImage)        # 由于输入的问题，输出的图片都是四维的，[0]
        y_hat = np.argmax(y_hat, axis=-1)
        print theta_1[0]
        showAnImage(transform1[0],mode='cv2')
        cv2.imwrite(dir+str(i)+'_1.png',transform1[0])
        print theta_2[0]
        showAnImage(transform2[0],mode='cv2')
        cv2.imwrite(dir+str(i) + '_2.png', transform2[0])
        print 'Ground truth is : {0}, the predicted label is : {1}'.format(y,y_hat)
        if y == y_hat:
            count += 1
        accu = count/n
    result =  'total estimate {0} images, the truth times is {1}, accu is: {2}'.format(n,count,accu)
    print result




eval_input(data['X_test'],data['y_test'],1340,20)
#eval_input(data['X_train'],data['y_train'],5360,10)

#print theta w b
