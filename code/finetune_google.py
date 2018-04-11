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
script_path = '/media/wangchen/newdata1/wangchen/work/Indoor/code'
sys.path.append(script_path)
from loadData import *
from net_googleNet import *
from function import *
from Conf import *

np.random.seed(123)


#global Variant
configFileURL = '/media/wangchen/newdata1/wangchen/work/Indoor/config/0212train_1.conf'
config = Conf(configFileURL)
learning_rate = config.getKeyValue('learning_rate')
batch_size = config.getKeyValue('batch_size')
test_batch_size = config.getKeyValue('test_batch_size')
num_epoches = config.getKeyValue('num_epoches')
learning_decay_step = config.getKeyValue('learning_decay_step')
learning_decay_mul = config.getKeyValue('learning_decay_mul')
weight_decay = config.getKeyValue('weight_decay')
extra = config.getKeyValue('extra')
model_url = config.getKeyValue('model_url')
resultFile_url = config.getKeyValue('resultFile_url')
reload_url = config.getKeyValue('reload_url')

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
net_prefix = 'googleNet/'



MyDict = makeDict(train_file_url, test_file_url)

print prefix.format(wangchendebug) + 'prepare image data finish!'
wangchendebug += 1


print prefix.format(wangchendebug) + 'reload params from caffemodel...'
wangchendebug += 1


copyParamsList = reloadModel(reload_url)


print prefix.format(wangchendebug) + 'build model...'
wangchendebug += 1

net_google_Dict = build_model(net_input, net_prefix, classificationFlag=True, lastClassNum=67, dropoutratio=0.5)
net_google_last_layer = net_google_Dict[net_prefix + 'prob']

lasagneLayerList = lasagne.layers.get_all_layers(net_google_last_layer)
lasagneParamList = lasagne.layers.get_all_params(net_google_last_layer, trainable=True)

lasagne.layers.set_all_param_values(net_google_last_layer, copyParamsList)
lasagneParamvaluesList = lasagne.layers.get_all_param_values(net_google_last_layer, trainable=True)
copyResult = checkParams(lasagneParamvaluesList,copyParamsList)

print prefix.format(wangchendebug) + 'build model and copy params finish!'
print '\tcopy params are: {0}'.format(copyResult[1])
wangchendebug += 1

X = T.tensor4()
y = T.ivector()

# training output
output_train = lasagne.layers.get_output([net_google_last_layer], X, deterministic=False)
output_train = output_train[0]
# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output([net_google_last_layer], X, deterministic=True)
output_eval = output_eval[0]
sh_lr = theano.shared(lasagne.utils.floatX(learning_rate))
l2_penalty = regularize_network_params(net_google_last_layer, l2)
cost = T.mean(T.nnet.categorical_crossentropy(output_train, y))
cost = cost + weight_decay*l2_penalty

updates = lasagne.updates.momentum(cost,lasagneParamList,sh_lr)


train = theano.function([X,y],[cost, output_train], updates=updates)
eval = theano.function([X], [output_eval], on_unused_input='warn')


def train_epoch(X, y):
    num_samples = X.shape[0]
    shuffle = np.random.permutation(num_samples)     # 用random函数，打乱数据
    num_batches = int(np.ceil(num_samples / float(batch_size)))
    costs = []
    correct = 0
    for i in range(num_batches):
        idx = range(i * batch_size, np.minimum((i + 1) * batch_size, num_samples))
        idx = shuffle[idx]
        X_batch = X[idx]
        y_batch = y[idx]
        cost_batch, output_train = train(X_batch, y_batch)
        costs += [cost_batch]
        preds = np.argmax(output_train, axis=-1)
        correct += np.sum(y_batch == preds)
    return np.mean(costs), correct / float(num_samples)


def eval_epoch(X, y):
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
        correct += np.sum(y_batch == preds)
    acc = correct / float(num_samples)
    return acc

writeInfo(resultFile_url, info)
loc_params = new_params = lasagne.layers.get_all_param_values(net_google_last_layer, trainable=True)
try:
    max_test_acc = 0.
    idx = 0

    for n in range(num_epoches):

        train_cost, train_acc = train_epoch(MyDict['X_train'], MyDict['y_train'])
        test_acc = eval_epoch(MyDict['X_test'], MyDict['y_test'])

        loc_params = new_params
        new_params = lasagne.layers.get_all_param_values(net_google_last_layer, trainable=True)
        if n >= 1:
            print checkParams(loc_params,new_params)

        if (n+1) % learning_decay_step == 0:
            new_lr = sh_lr.get_value() * learning_decay_mul
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))

        currentInfo = "Epoch {0}: Train cost {1}, Train acc {2}, test acc {3}".format(
                n, train_cost, train_acc, test_acc)
        writeInfo(resultFile_url, currentInfo + '\n')
        print currentInfo

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            idx = n+1
            saveModel(model_url, lasagne.layers.get_all_param_values(net_google_last_layer, trainable = True))
            currentInfo =  "the model has been saved !"
            writeInfo(resultFile_url, currentInfo + '\n')

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


