# -*- coding: UTF-8 -*-

##### 一次将图片加载至内存

import cv2
import numpy as np
import lasagne

from sklearn import preprocessing

#the global variables
train_num = 5360
test_num = 1340
labeldata = ["airport_inside","artstudio","auditorium","bakery","bar",
                  "bathroom","bedroom","bookstore","bowling","buffet",
                  "casino","children_room","church_inside","classroom","cloister",
                  "closet","clothingstore","computerroom","concert_hall",
                  "corridor","deli","dentaloffice","dining_room","elevator",
                  "fastfood_restaurant","florist","gameroom","garage","greenhouse",
                  "grocerystore","gym","hairsalon","hospitalroom","inside_bus",
                  "inside_subway","jewelleryshop","kindergarden","kitchen","laboratorywet",
                  "laundromat","library","livingroom","lobby","locker_room","mall",
                  "meeting_room","movietheater","museum","nursery","office","operating_room",
                  "pantry","poolinside","prisoncell","restaurant","restaurant_kitchen",
                  "shoeshop","stairscase","studiomusic","subway","toystore",
                  "trainstation","tv_studio","videostore","waitingroom",
                  "warehouse","winecellar"]

file_root = '/home/rootcs412/wangchen/dataSet/Indoor67/Images/'
train_file_url = '/home/rootcs412/wangchen/dataSet/Indoor67/Images/TrainImages.txt'
test_file_url = '/home/rootcs412/wangchen/dataSet/Indoor67/Images/TestImages.txt'

def getLabel(label_str):
    for i in range(len(labeldata)):
        if label_str == labeldata[i]:
            return i
    return -1

def loadImage(imageURL):
    ima = cv2.imread(imageURL)
    ima = cv2.resize(ima, (224, 224))  # 图像像素调整 ——》224*224
    ima = np.asarray(ima, dtype='float32') / 255.
    #cv2.cvtColor()
    ima = ima.transpose(2, 0, 1)  # 这张图片的格式为(h,w,rgb), 然后想办法交换成(rgb,h,w)
    return ima

def loadImageFromFile(fileURL):
    data = []
    label = []
    file = open(fileURL,'r')
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip()
        pos = line.find('/')
        label_str = line[:pos]
        curLabel = getLabel(label_str)
        if curLabel == -1:
            raise ValueError('label or url error!')
        image_url = file_root + line
        curIma = loadImage(image_url)    # 加载单张图片，用append方法拼成数组
        data.append(curIma)
        label.append(curLabel)
    return data,label

def scaleData(train_data, test_data):
    train_data = np.array(train_data).ravel()
    train_data = train_data.reshape(train_data.shape[0],1)
    test_data = np.array(test_data).ravel()
    test_data = test_data.reshape(test_data.shape[0],1)
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    train_data = np.reshape(train_data,(train_num,3,224,224))
    test_data = np.reshape(test_data,(test_num,3,224,224))
    return train_data,test_data

def makeDict(train_file_url, test_file_url):
    trainData, trainLabel = loadImageFromFile(train_file_url)  # list list
    testData, testLabel = loadImageFromFile(test_file_url)  # list list
    trainData, testData = scaleData(trainData, testData)  # ndarray ndarray
    X_train = lasagne.utils.floatX(trainData)
    y_train = np.array(trainLabel)
    y_train = y_train.astype('int32')
    X_test = lasagne.utils.floatX(testData)
    y_test = np.array(testLabel)
    y_test = y_test.astype('int32')
    return dict(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)


if __name__ == '__main__':
    '''
    trainData, trainLabel = loadImageFromFile(train_file_url)
    testData, testLabel = loadImageFromFile(test_file_url)
    trainData, testData = scaleData(trainData,testData)
    '''
    data = makeDict(train_file_url,test_file_url)
    print 'check finish!'

