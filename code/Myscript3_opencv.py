# -*-coding: UTF-8 -*-

import numpy as np
import cv2
from math import *


def rotate4(image, angle):
    (h, w) = image.shape[:2]
    anglePi = angle*math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)
    X1 = abs(0.5*h*cosA + 0.5*w*sinA)
    X2 = abs(0.5*h*cosA - 0.5*w*sinA)
    Y1 = abs(-0.5*h*sinA + 0.5*w*cosA)
    Y2 = abs(-0.5*h*sinA - 0.5*w*cosA)
    H = 2*max(Y1,Y2)
    W = 2*max(X1,X2)
    scale1 = H/h
    scale2 = W/w
    scale = max(scale1,scale2)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated



url = '/home/wangchen/DataSet/Indoor67/Images/corridor/corridor04.jpg'

image = cv2.imread(url)

cv2.imshow('0',image)

image1 = cv2.resize(image,(224,224))
cv2.imshow('1',image1)

'''
image1 = rotate(image,45)
cv2.imshow('1',image1)
'''


image4 = rotate4(image,-45)
image4 = cv2.resize(image4,(224,224))
cv2.imshow('4',image4)
cv2.waitKey(0)
