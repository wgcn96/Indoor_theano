# -*- coding: UTF-8 -*-

import collections
import string

class Conf:
    def __init__(self,configURL):
        self.configURL = configURL
        self.Dict = dict([('extra','')])
        with open(self.configURL,'r') as configFile:
            while True:
                line = configFile.readline()
                line = line.strip()
                if line == '':
                    break
                pos = line.find('=')
                key = line[:pos]
                value = line[pos+1:]
                key = key.strip()
                value = value.strip()
                self.Dict[key] = value
        configFile.close

    def getKeyValue(self,keyValue):
        if keyValue == 'extra' or keyValue == 'model_url' or keyValue == 'resultFile_url' or keyValue == 'reload_url' or keyValue == 'matrixTable_url' or keyValue == 'detailFile_url':
            return self.Dict[keyValue]      #string
        elif keyValue == 'learning_rate' or keyValue == 'learning_decay_mul' or keyValue == 'weight_decay' or keyValue == 'stn_lr_mul':
            return string.atof(self.Dict[keyValue])         #float
        else:
            return string.atoi(self.Dict[keyValue])         #int


if __name__ == '__main__':
    config = Conf('/media/wangchen/newdata1/wangchen/work/Indoor/config/1219train_1.conf')
    print config.Dict