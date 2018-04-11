# -*- coding: UTF-8 -*-

import os
os.environ['THEANO_FLAGS'] = "device=cuda1"
import theano
import lasagne
import pickle
import theano.tensor as T
import numpy as np
import datetime
from lasagne.regularization import *
from collections import OrderedDict

import sys
script_path = '/home/rootcs412/wangchen/work/Indoor/code'
sys.path.append(script_path)
from loadData_aug4 import *
from net_vggWithSTN import *
from function import *
from Conf import *



# Hyperparamters && global Variant
configFileURL = '/home/rootcs412/wangchen/work/Indoor/config/0121train_1.conf'
config = Conf(configFileURL)
learning_rate = config.getKeyValue('learning_rate')
batch_size = config.getKeyValue('batch_size')
test_batch_size = config.getKeyValue('test_batch_size')
num_epoches = config.getKeyValue('num_epoches')
learning_decay_step = config.getKeyValue('learning_decay_step')
learning_decay_mul = config.getKeyValue('learning_decay_mul')
stn_lr_mul = config.getKeyValue('stn_lr_mul')
weight_decay = config.getKeyValue('weight_decay')
extra = config.getKeyValue('extra')
model_url = config.getKeyValue('model_url')
resultFile_url = config.getKeyValue('resultFile_url')
matrixTable_url = config.getKeyValue('matrixTable_url')

reload_image_url = '/home/rootcs412/wangchen/work/Indoor/models/caffe_vgg16_image.pkl'
reload_places_url = '/home/rootcs412/wangchen/work/Indoor/models/caffe_vgg16_places.pkl'

# for record
info = ''
startTime = datetime.datetime.now()
info += '\nstart time is: ' + startTime.strftime('%Y-%m-%d %H:%M:%S') +'\n'
info += '\nthis experiment hyperparameters are:\n'
info += 'config_file :' + configFileURL + '\n'
info += 'learning_rate = ' +str(learning_rate) + '\n'
info += 'batch_size = '+str(batch_size) + '\n'
info += 'num_epoches = ' +str(num_epoches) + '\n'
info += 'model_file :' + model_url +'\n'
##### if there are any things to record, write to str
info += '\n其余要说明的事项 ：'
extra += ''
info += extra +'\n'
info += '\n'


# for debug
wangchendebug = 1
prefix = 'My debug output {}: '
#net_prefix = 'vgg/'



MyDict = makeDict(train_file_url, test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1


print prefix.format(wangchendebug) + 'reload params from caffemodel...'
wangchendebug += 1


copyImageParamsList = reloadModel(reload_image_url)
copyPlacesParamsList = reloadModel(reload_places_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1

net_final_out,net_global_out,net_inc1_out,net_inc2_out = buildSpatialTransformerNet()
lasagneLayersList = lasagne.layers.get_all_layers(net_final_out)
lasagneParamsList = lasagne.layers.get_all_params(net_final_out, trainable=True)

count1 = copyParams(lasagneLayersList[1:22],copyPlacesParamsList)
count2 = copyParams(lasagneLayersList[23:41],copyPlacesParamsList)
count3 = copyParams(lasagneLayersList[44:65],copyPlacesParamsList)
count4 = copyParams(lasagneLayersList[68:89],copyImageParamsList)



print prefix.format(wangchendebug) + 'build model and copy params finish!'
print '\tcopy params are: {} {} {} {} '.format(count1,count2,count3,count4)
wangchendebug += 1

X = T.tensor4()
y = T.ivector()

# training output
output_train,global_output,inc1_output,inc2_output = lasagne.layers.get_output([net_final_out,net_global_out,net_inc1_out,net_inc2_out], X, deterministic=False)

# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(net_final_out, X, deterministic=True)

sh_lr = theano.shared(lasagne.utils.floatX(learning_rate))
l2_penalty = regularize_network_params(net_final_out,l2)
cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
#cost = cost + weight_decay*l2_penalty

global_cost = T.mean(T.nnet.categorical_crossentropy(global_output,y))
inc1_cost = T.mean(T.nnet.categorical_crossentropy(inc1_output,y))
inc2_cost = T.mean(T.nnet.categorical_crossentropy(inc2_output,y))
cost = cost + 0.1 * global_cost + weight_decay *l2_penalty
'''
grads = []
for param in lasagneParamList:
    grad = T.grad(cost,param)
    grads.append(grad)
updates = OrderedDict()


for param,grad in zip(lasagneParamList[:30], grads[:30]):
    updates[param] = param - sh_lr * grad
for param,grad in zip(lasagneParamList[30:], grads[30:]):
    updates[param] = param - 2 * sh_lr * grad
'''

updates = lasagne.updates.sgd(cost,lasagneParamsList[:30],sh_lr)
updates.update(lasagne.updates.sgd(cost,lasagneParamsList[30:60],stn_lr_mul*sh_lr))
updates.update(lasagne.updates.sgd(cost,lasagneParamsList[60:90],sh_lr))
updates.update(lasagne.updates.sgd(cost,lasagneParamsList[90:92],stn_lr_mul*sh_lr))
updates.update(lasagne.updates.sgd(cost,lasagneParamsList[92:],sh_lr))


train = theano.function([X,y],[cost, output_train,global_cost,inc1_cost,inc2_cost], updates=updates)
eval = theano.function([X], [output_eval], on_unused_input='warn')


def train_epoch(X, y):
    num_samples = X.shape[0]
    shuffle = np.random.permutation(num_samples)     # 用random函数，打乱数据
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    costs = []
    global_cost = []
    inc1_cost = []
    inc2_cost = []
    correct = 0
    for i in range(num_batches):
        idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_samples))
        idx = shuffle[idx]
        X_batch = X[idx]
        y_batch = y[idx]
        cost_batch, output_train,global_cost_batch,inc1_cost_batch,inc2_cost_batch = train(X_batch, y_batch)
        costs += [cost_batch]
        global_cost += [global_cost_batch]
        inc1_cost += [inc1_cost_batch]
        inc2_cost += [inc2_cost_batch]
        preds = np.argmax(output_train, axis=-1)
        correct += np.sum(y_batch == preds)
    return np.mean(costs), correct / float(num_samples),np.mean(global_cost),np.mean(inc1_cost),np.mean(inc2_cost)


def eval_epoch(X, y):
    test_label = []
    y_hat = []
    num_samples = X.shape[0]
    shuffle = np.random.permutation(num_samples)
    num_batches = int(np.ceil(num_samples / float(test_batch_size)))
    correct = 0
    for i in range(num_batches):
        idx = range(i * test_batch_size, np.minimum((i + 1) * test_batch_size, num_samples))
        idx = shuffle[idx]
        X_batch = X[idx]
        y_batch = y[idx]
        output_eval = eval(X_batch)
        preds = np.argmax(output_eval, axis=-1)
        preds = preds.flatten()
        correct += np.sum(y_batch == preds)
        test_label.extend(y_batch)
        y_hat.extend(preds)
    acc = correct / float(num_samples)
    return acc,test_label,y_hat

writeInfo(resultFile_url, info)
loc_params = new_params = lasagne.layers.get_all_param_values(net_final_out,trainable=True)
try:
    max_test_acc = 0.
    idx = 0

    for n in range(num_epoches):

        train_cost, train_acc,global_cost,inc1_cost,inc2_cost = train_epoch(MyDict['X_train'], MyDict['y_train'])
        test_acc , test_label, y_hat = eval_epoch(MyDict['X_test'], MyDict['y_test'])

        loc_params = new_params
        new_params = lasagne.layers.get_all_param_values(net_final_out,trainable=True)
        if n >= 1:
            print checkParams(loc_params,new_params)

        if (n+1) % learning_decay_step == 0:
            new_lr = sh_lr.get_value() * learning_decay_mul
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

        currentInfo = "Epoch {0}: Train cost {1}, Train acc {2}, test acc {3}".format(
                n, train_cost, train_acc, test_acc)
        #extraInfo =currentInfo + "\nglobal cost {0}, inc1 cost {1}, inc2 cost {2}".format(global_cost,inc1_cost,inc2_cost)
        writeInfo(resultFile_url, currentInfo + '\n')
        print currentInfo

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            idx = n+1
            matrixTabel, lineDetail = confusionMatrix(test_label, y_hat)
            matrixToExcel(matrixTabel, matrixTable_url)
            #detailToFile(lineDetail,detailFile_url)
            saveModel(model_url,lasagne.layers.get_all_param_values(net_final_out,trainable = True))
            currentInfo =  "the model has been saved !"
            writeInfo(resultFile_url, currentInfo + '\n')
            print currentInfo

    result = "we get best accuracy {0} , at the {1} epoch\n".format(max_test_acc,idx)
    result += "the model has been saved !"
    info = ''
    info += result+'\n'
    endTime = datetime.datetime.now()
    info += 'experiment end time is: ' + endTime.strftime('%Y-%m-%d %H:%M:%S') + '\n'
    totalSeconds = (endTime-startTime).seconds
    timeInfo =  'the program run total {} seconds .'.format(totalSeconds)
    info += timeInfo
    print result
    print timeInfo
    writeInfo(resultFile_url, info)
except KeyboardInterrupt:
    pass


