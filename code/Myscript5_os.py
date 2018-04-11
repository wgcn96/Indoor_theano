# -*- coding: UTF-8 -*-

import os

a,b=os.path.split("winecellar/bodega_12_11_flickr.jpg")
print a
print b

c,d=os.path.splitext("winecellar/bodega_12_11_flickr.jpg")
print c
print d


currentImageURL_NoRoot = "winecellar/bodega_12_11_flickr.jpg"
pos = currentImageURL_NoRoot.find('/')
imageClass = currentImageURL_NoRoot[:pos]
imageName = currentImageURL_NoRoot[pos + 1:-4]