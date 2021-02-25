# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:08:04 2021

@author: Max
"""
import cv2, skimage.feature
import numpy as np
from sklearn.metrics.cluster import entropy
import matplotlib.pyplot as plt

def get_dictArrCoeffs(img, step, paramList=['contrast']):
    """
    Вычисление текстурных коэффициентов доменных блоков изображения
    
    Parameters
    ----------
    img : numpy array
        Массив изображения в градациях серого
    step : int
        Размер доменного блока
    paramList : list, optional
        Список вычисляемых птекстурных коэффициентов. The default is ['contrast'].

    Returns
    -------
    coefs : Dict
        Словарь массивов вычисленных коэффициентов.

    """
    coefs = dict.fromkeys(paramList)
    h,l=img.shape
    for p in paramList:
        t = np.empty((h//step, l//step))
        x0,y0=0,0
        for x in range(0,l,step):
            y0=0
            for y in range(0,h,step):
                u=img[y:y+step,x:x+step]
                t[y0,x0]=get_allCoeffs(u, param=p)
                y0+=1
            x0+=1
        coefs[p]=t.copy()
    return coefs

def get_allCoeffs(img,param='contrast'):
    """
    Вычисление текстурного коэффициента

    Parameters
    ----------
    img : numpy array
        доменный блок изображения в градациях серого.
    param : str, optional
        Название вычисляемого коэффициента. The default is 'contrast'.

    Returns
    -------
    float
        значение вычисленного коэффициента.

    """
    if param=='entropy':
        return entropy(img)
    else:
        g = skimage.feature.greycomatrix(img, distances=[2], angles=[0], levels=256, normed=True, symmetric=True)
        return skimage.feature.greycoprops(g, param)[0, 0]
    


if __name__ == '__main__':
    """Пример пользования функции """
    #Размеры изображений д.б. степень 2
    imgFileName = 'd:\\CV\\22\\201395601\\10\\00000XXX\\P6740010_00526534_10_srcimg_0045.tif'
    img = cv2.cvtColor(cv2.imread(imgFileName), cv2.COLOR_BGR2GRAY)
    
    #Список вычисляемых текстурных коэффициентов
    listCoeffs=['contrast', 
                'entropy', 
                'energy', 
                'dissimilarity', 
                'homogeneity', 
                'ASM', 
                'correlation']
    
    #Размер доменного блока. ОБЯЗАТЕЛЬНО СТЕПЕНЬ 2
    domSize=16
    
    t=get_dictArrCoeffs(img, domSize, paramList=listCoeffs)
    
    """Построение тепловой карты текстурных коэффициентов"""
    fig, axs = plt.subplots(len(listCoeffs)+1)
    i=0
    for r in listCoeffs:
        axs[i].imshow(t[r], cmap='hot')
        axs[i].set_title(r)
        i+=1
    axs[i].imshow(img, cmap='gray')