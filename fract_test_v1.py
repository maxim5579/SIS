# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 19:42:56 2020

@author: Max
"""

import cv2, os, skimage, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def resizer(src, percent=100,siz=(0,0)):
    if siz==(0,0):
        width = int(src.shape[1] * percent / 100)
        height = int(src.shape[0] * percent / 100)
        return cv2.resize(src, (width, height), interpolation = cv2.INTER_AREA)
    else:
        return cv2.resize(src, (siz[1], siz[0]), interpolation = cv2.INTER_AREA)


def binarization(arr,f=False):
    return cv2.adaptiveThreshold(arr,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,11)
#    if f==True:
#        t=resizer(arr,60)
#        th3 = cv2.adaptiveThreshold(t,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,11)
#        return resizer(th3,siz=arr.shape)
#    else:        
#        return cv2.adaptiveThreshold(arr,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,11)

def morphOp(img):
    bm=img.copy()
    L = skimage.measure.label(bm,background=1)
    props = skimage.measure.regionprops(L)    
    z = [n for n in props if n.area<=5] # 28
    mask=np.zeros(L.shape)
    for i in z:
        tmp=L==i.label
        L[tmp]=0        
    mask=L==0
    bm[mask]=1    
    return bm

def ROIzer(img):
    mask_bi=morphOp(binarization(img))
    props = skimage.measure.regionprops(skimage.measure.label(mask_bi,background=1))
    t=[i.bbox for i in props]
    r=1
    for i in t:
        img = cv2.rectangle(img,(i[1]-r,i[0]-r),(i[3]+r,i[2]+r),(255,255,255),1)
    return img

if __name__ == '__main__':
    patch = 'd:\\CV\\22\\201385302\\10\\00000XXX\\'
    file = 'P6740010_00525844_10_srcimg_0006.tif'
#    isIm=cv2.cvtColor(cv2.imread(patch+file), cv2.COLOR_BGR2GRAY)
    
    allFotos = os.listdir(patch+'/')
    
    imgs=[]
    a,c=5,5    
    for i in tqdm(allFotos[a:a+c]):
        isIm=cv2.cvtColor(cv2.imread(patch+i), cv2.COLOR_BGR2GRAY)
#        isIm=cv2.equalizeHist(isIm)
        imgs.append(ROIzer(isIm))
#    cv2.imshow('test', img)  
        
    fig, axs = plt.subplots(len(imgs))
    j=0
    for i in imgs:
        axs[j].imshow(i, cmap='gray')
        j+=1