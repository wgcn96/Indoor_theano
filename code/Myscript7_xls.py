# -*- coding: UTF-8 -*-

import xlwt
import xlrd
from loadData import labeldata

ori_xls_url = '/media/wangchen/newdata1/wangchen/work/Indoor/result/FEATURE9.xls'
new_xls_url = '/media/wangchen/newdata1/wangchen/work/Indoor/result/newFEATURE9.xls'

oriworkbook = xlrd.open_workbook(ori_xls_url)
orisheet = oriworkbook.sheet_by_index(0) # sheet索引从0开始
newworkbook = xlwt.Workbook()
newsheet = newworkbook.add_sheet('sheet0')

def getTabel():
    table = []
    for i in range(67):
        currentRow = orisheet.row_values(i)
        table.append(currentRow)

    return table

def writeNewTabel(tabel):
    for i in range( 67 ):
        newsheet.write(0,i+1,labeldata[i])
        newsheet.write(i + 1, 0, labeldata[i])
    for i in range( len(tabel)):
        for j in range( len(tabel[i]) ):
            if tabel[i][j] != 0:
                newsheet.write(i+1,j+1,tabel[i][j])
    newworkbook.save(new_xls_url)

def main():
    tabel = getTabel()
    writeNewTabel(tabel)

if __name__ == '__main__':
    main()