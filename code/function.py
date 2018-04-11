# -*- coding: UTF-8 -*-

import cPickle as pickle
#import pickle
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import cv2
import xlwt

from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import DenseLayer

from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA


from loadData import labeldata


# pickle module start
def reloadModel(url):
    with open(url, 'rb') as f:
        paramList = pickle.load(f)
    f.close()
    return paramList


def saveModel(url, params):
    with open(url, 'wb') as f:
        pickle.dump(params, f, protocol=1)
    f.close()


# pickle module end


# lasagne module start
def copyParams(lasagneLayerList, oriParamList):
    n = len(lasagneLayerList)
    pos = 0
    for i in range(n):
        currentLayer = lasagneLayerList[i]
        currentParamList = currentLayer.get_params(trainable=True)
        if currentParamList != []:
            for param in currentParamList:
                param.set_value(oriParamList[pos])
                pos += 1
    return pos  # 处理的參數數量


def copyParams_bn(lasagneLayerList, oriParamsList):
    count = pos = 0
    for layer in lasagneLayerList:
        # print 'current layer:' + str(layer.name)
        if isinstance(layer, (ConvLayer, DenseLayer)):
            layer.W.set_value(oriParamsList[pos])
            pos += 2
            count += 1
    return count  # 处理的參數數量


def checkParams(valueList1, valueList2):
    resultCounter = 0
    for i in range(len(valueList1)):
        param1 = valueList1[i]
        param2 = valueList2[i]
        if (param1 == param2).all() == True:
            resultCounter += 1
    if resultCounter == 0:
        return True, resultCounter
    else:
        return False, resultCounter

# lasagne module end


# image module start
def showAnImage(imageData,mode='cv2',label='0',waitKey=0):
    revortedImage = cv2.normalize(imageData.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    revortedImage = np.transpose(revortedImage,(1,2,0))
    #revortedImage = cv2.cvtColor(revortedImage, cv2.COLOR_BGR2RGB)
    if mode == 'cv2':
        cv2.imshow(label,revortedImage)
        cv2.waitKey(waitKey)
        cv2.destroyAllWindows()
    else:
        plt.imshow(revortedImage)
        plt.show()

# image module end

# sklearn module start
def myPCA(train_feature, test_feature, component):
    pca = PCA(n_components=component)
    scaler = preprocessing.StandardScaler().fit(train_feature)
    train_feature_scale = scaler.transform(train_feature)
    test_feature_scale = scaler.transform(test_feature)
    pca.fit(train_feature_scale)
    train_feature_pca = pca.transform(train_feature_scale)
    test_feature_pca = pca.transform(test_feature_scale)
    return train_feature_pca,test_feature_pca

def mySVM(train_data,test_data,train_label,test_label):
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data,train_label)
    y_hat = clf.predict(test_data)
    m_accuracy = metrics.accuracy_score(test_label,y_hat)
    return m_accuracy

def mySVM2(train_data,test_data,train_label,test_label):
    clf = svm.SVC(kernel='poly')
    clf.fit(train_data,train_label)
    y_hat = clf.predict(test_data)
    m_accuracy = metrics.accuracy_score(test_label,y_hat)
    return m_accuracy

def mySVM3(train_data,test_data,train_label,test_label):
    clf = svm.SVC(kernel='rbf')
    clf.fit(train_data,train_label)
    y_hat = clf.predict(test_data)
    m_accuracy = metrics.accuracy_score(test_label,y_hat)
    return m_accuracy

def mySVM4(train_data,test_data,train_label,test_label):
    clf = svm.SVC(kernel='sigmoid')
    clf.fit(train_data,train_label)
    y_hat = clf.predict(test_data)
    m_accuracy = metrics.accuracy_score(test_label,y_hat)
    return m_accuracy
# sklearn module end

# result module start
def detail_result(test_label,y_hat):
    m_accuracy = metrics.accuracy_score(test_label,y_hat)
    m_recall = metrics.recall_score(test_label,y_hat,average='macro')
    m_f1score = metrics.f1_score(test_label,y_hat,average='macro')
    return m_accuracy,m_recall,m_f1score

def confusionMatrix(test_label,y_hat):
    detailList = []
    matrixTabel = np.zeros([67,67])
    for i in range( len(test_label)):
        if y_hat[i] != test_label[i]:
            tmp = [i+1, test_label[i], y_hat[i]]
            detailList.append(tmp)
            matrixTabel[test_label[i],y_hat[i]] += 1
    detailList = np.asarray(detailList,dtype='int32')
    return matrixTabel,detailList

def detailToFile(numpy_data, outfileurl):
    #np.savetxt(outfileurl, numpy_data, fmt='%2.4f')
    np.savetxt(outfileurl, numpy_data)

def matrixToExcel(matrixTable, xlsurl):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')
    for i in range( 67 ):
        sheet.write(0,i+1,labeldata[i])
        sheet.write(i + 1, 0, labeldata[i])
    for i in range(len(matrixTable)):
        for j in range(len(matrixTable)):
            if matrixTable[i,j] != 0:
                sheet.write(j+1, i+1, matrixTable[i, j])
    wbk.save(xlsurl)

def writeInfo(fileURL,info):
    with open(fileURL,'a') as f:
        f.write(info)
    f.close()

# result module end

'''
# caffe module start
caffe_root = '/home/wangchen/last_caffe_with_stn-master/python'
import sys
sys.path.insert(0,caffe_root)
import caffe
#print caffe.__path__


def getOriParams(net_prototxt,net_weight):
    net_ori = caffe.Net(net_prototxt, net_weight, caffe.TEST)
    paramList = []
    for param_name in net_ori.params.keys():
        weight = net_ori.params[param_name][0].data
        bias = net_ori.params[param_name][1].data
        paramList.append(weight)
        paramList.append(bias)
        #print param_name,weight,bias
    return paramList


def checkDimes(paramsList1_theano, paramsList2_caffe):
    print 'paramsList1.dimes is: {0}'.format(len(paramsList1_theano))
    print 'paramsList2.dimes is: {0}'.format(len(paramsList2_caffe))
    for i in range(len(paramsList1_theano)):
        result = ''
        #result +=  paramsList1_theano[i].name
        theano_shape = paramsList1_theano[i].shape
        caffe_shape = paramsList2_caffe[i].shape
        if theano_shape == caffe_shape:
            result += '\tTrue'
        else:
            result += '\tFalse'
            result += '\n'
            result += str(i) + '\t'
            result += 'caffe_dimes is: {0} ,target_dims is: {1}'.format(caffe_shape,theano_shape)
        print result

def caffeFCTransform(caffeParamList,layerNumberList):
    for i in layerNumberList:
        caffeParamList[i] = caffeParamList[i].T

def convertCaffeMeanFile(meanFileURL, npyFileURL):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(meanFileURL, 'rb').read()
    blob.ParseFromString(bin_mean)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    npy_mean = arr[0]
    np.save(npyFileURL, npy_mean)
    print 'finish!'
    
# caffe module end
'''

if __name__ == '__main__':
    #npyFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.npy'
    #meanFileURL = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/birds-mean.binaryproto'
    #convertCaffeMeanFile(meanFileURL,npyFileURL)

    '''
    info1 = 'info1'
    info2 = 'info2'
    testtxt = '/media/wangchen/newdata1/wangchen/work/CUB-Birds-googleNet/models/test.txt'
    writeInfo(testtxt,info1)
    writeInfo(testtxt,info2)
    '''

    '''
    prototxt = '/media/wangchen/data/wangchen/Indoor/models/lsq_PlaceModel_14th/deploy.prototxt'
    weight = '/media/wangchen/data/wangchen/Indoor/models/lsq_PlaceModel_14th/vgg.caffemodel'
    params = getOriParams(prototxt,weight)
    '''

    '''
    a = np.asarray([0,1,3,5,7,9])
    b = np.asarray([0,2,4,6,8,9])
    r1, r2 = confusionMatrix(a,b)
    print r1,r2
    '''
    print 'finish!'
