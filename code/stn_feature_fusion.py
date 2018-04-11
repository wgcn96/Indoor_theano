# -*- coding: UTF-8 -*-


from dataBase import *
#from loadData import *
from function import *

from sklearn.neural_network import MLPClassifier

import datetime

db,cursor = connectdb()

#data = makeDict(train_file_url,test_file_url)

startTime = datetime.datetime.now()

test_feature , y_test = read_feature(db,cursor,testtable,'FEATURE1',1340,labelFlag=True)
train_feature , y_train = read_feature(db,cursor,traintable,'FEATURE1',5360,labelFlag=True)

endTime = datetime.datetime.now()
totalSeconds = (endTime - startTime).seconds

print 'total load feature time :',totalSeconds


startTime = datetime.datetime.now()

m_precision_1 = mySVM(train_feature,test_feature,y_train,y_test)
m_precision_2 = mySVM2(train_feature,test_feature,y_train,y_test)
m_precision_3 = mySVM3(train_feature,test_feature,y_train,y_test)
m_precision_4 = mySVM4(train_feature,test_feature,y_train,y_test)

train_feature_pca,test_feature_pca = myPCA(train_feature, test_feature, 0.99)
m_precision_1_2 = mySVM(train_feature_pca,test_feature_pca,y_train,y_test)
m_precision_2_2 = mySVM2(train_feature_pca,test_feature_pca,y_train,y_test)
m_precision_3_2 = mySVM3(train_feature_pca,test_feature_pca,y_train,y_test)
m_precision_4_2 = mySVM4(train_feature_pca,test_feature_pca,y_train,y_test)

print m_precision_1,m_precision_1_2
print m_precision_2,m_precision_2_2
print m_precision_3,m_precision_3_2
print m_precision_4,m_precision_4_2

endTime = datetime.datetime.now()
totalSeconds = (endTime - startTime).seconds

print 'total load feature time :',totalSeconds


'''
scaler = preprocessing.StandardScaler().fit(train_feature)
train_feature_scale = scaler.transform(train_feature)
test_feature_scale = scaler.transform(test_feature)
clf = MLPClassifier(hidden_layer_sizes=(67,),batch_size=16,learning_rate_init=0.002,solver='sgd',activation='identity',learning_rate='adaptive')
clf.fit(train_feature,y_train)

y_hat = clf.predict(test_feature)

count = 0.
for i in range(1340):
    if y_hat[i] == y_test[i]:
        count += 1
accu = count/1340
m_accuracy = metrics.accuracy_score(y_test,y_hat)

print m_accuracy,accu
'''
closedb(db)