# -*- coding: UTF-8 -*-

import string

file1 = '/media/wangchen/newdata1/wangchen/work/copy/result/0105result_1.txt'
file2 = '/media/wangchen/newdata1/wangchen/work/copy/result/0105result_2.txt'

def find_last(string, str):
    last_position=-1
    while True:
        position=string.find(str, last_position+1)
        if position==-1:
            return last_position
        last_position=position

def getMaxAccu(file_url):
    file = open(file_url,'a+')
    epoch = 0
    accu = 0.
    epochpos = 0
    while True:
        line = file.readline()
        if not line:
            break
        line = line.strip()
        accuExitFlag = line.find('Epoch')
        if accuExitFlag > -1:
            print epoch
            epoch += 1
            pos = line.find('test acc ')
            curaccustr = line[pos+8:]
            curaccu = string.atof(curaccustr)
            if curaccu > accu:
                accu = curaccu
                epochpos = epoch
    file.write('the best accuracy is : {0} ,at the {1} epoch\n'.format(accu,epochpos))
    file.close()
    return accu,epochpos

if __name__ == "__main__":
    r1,r2 = getMaxAccu(file1)
    print r1
    print r2