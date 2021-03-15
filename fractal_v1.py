# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:36:53 2020

@author: Max
"""

import cv2, os
from skimage.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

def get_window(img,t_s=110):
    h,l=img.shape
    return isIm[:h,t_s:t_s+h]

class FractalModel():
    def __init__(self, img, max_levels=4, min_error=100, fileDomens=''):
        self.img = img
        self.minError = min_error
        self.max_level=max_levels
        
        self.domens=[]
        if fileDomens!='' and os.path.exists(fileDomens)==True:
            self.get_domens1(fileDomens)
        else:
            self.get_domens(img.copy(), d_size=16)
            
        self.di_tmp={}
        self.allRes=pd.DataFrame(columns = ['rang_id', 'domen_id', 'Scale', 'Contrast', 'Brigth', 'Error'])
        self.DivideRang((0,0,img.shape[1],img.shape[0]),'',1)
    
    def DivideRang(self,coord_tuple, parentID, level):
        if level<=self.max_level and self.__calculateError__(parentID,coord_tuple)==False:
            coords=self.__divider__(coord_tuple[0],coord_tuple[1],coord_tuple[2],coord_tuple[3])
            for i in range(4):
                self.di_tmp[parentID+str(i+1)]=coords[i]
                self.DivideRang(coords[i],parentID+str(i+1),level+1)
    
    def __divider__(self,x1,y1,x2,y2):
        coords=[]
        c_x,c_y=(int(x1+(x2-x1)/2),int(y1+(y2-y1)/2))
        coords.append((x1,y1,c_x,c_y))
        coords.append((c_x,y1,x2,c_y))
        coords.append((c_x,c_y,x2,y2))
        coords.append((x1,c_y,c_x,y2))
        return coords
    
    def __calculateError__(self,rang_ID, rangCoord):
        rang=self.img[rangCoord[0]:rangCoord[2],rangCoord[1]:rangCoord[3]].copy()
        ind_dom=0
        best_res=dict.fromkeys(['rang_id', 'domen_id', 'Scale', 'Contrast', 'Brigth', 'Error'])
        for dom in self.domens:
            res=self.CalculateCoeff(dom.copy(),rang.copy())
            if best_res['Error']==None or res['Error']<best_res['Error']:
                best_res={'rang_id':rang_ID,
                          'domen_id':ind_dom,
                          'Scale':res['Scale'],
                          'Contrast':res['Contrast'],
                          'Brigth':res['Brigth'],
                          'Error':res['Error']
                        }
            ind_dom+=1
        
        if best_res['Error']!=None and best_res['Error']<=self.minError:
            self.allRes.loc[self.allRes.shape[0]]=best_res
            return True
        else:
            return False
    
    def get_domens1(self, filename):
        img=cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        for i in range(0,img.shape[0],img.shape[1]):
            self.domens.append(img[i:i+img.shape[1], :])

    
    def get_domens(self, img, d_size=16):
        #Подготовка доменов
        h,l=img.shape
        for i in range(0,l-d_size,d_size+1):
            for j in range(0,h-d_size,d_size+1):
                self.domens.append(img[j:j+d_size,i:i+d_size].copy())
    
    def ContrastAndBrightness(self,img,c, b):
        tmp=img*c+b
#    tmp[tmp<0]=0
#    tmp[tmp>255]=255
        return tmp

    def ComputeError(self,domen, rang):
        # return np.sum((domen-rang)**2)/(rang.shape[0]*rang.shape[0])
        # return np.sqrt(np.sum((domen-rang)**2))/(rang.shape[0]*rang.shape[0])*100
#        return np.std(domen-rang)
        return mse(domen,rang)
    
    def ComputeError2(self,domen, rang):
        domen=np.uint8(domen)
        rd = cv2.resize(domen, (16,16), interpolation = cv2.INTER_AREA)
        rr=cv2.resize(rang, (16,16), interpolation = cv2.INTER_AREA)
        hist1 = cv2.calcHist([rd-np.min(rd)],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([rr-np.min(rr)],[0],None,[256],[0,256])
        dh=hist1-hist2 
        r=np.sqrt((hist1-hist2)**2).mean()
        # r=mse(hist1,hist2)
        return r
    
    def FindSQ(self,domen, rang):
        alpha, beta=0,0
        alpha=np.sum((domen-domen.mean())*(rang-rang.mean()))
        beta=np.sum(np.power(domen-domen.mean(),2))
        if beta==0:
            beta=1
        return (alpha/beta, rang.mean()-alpha/beta*domen.mean())

    def CalculateCoeff(self,domen,rang):
        #Масштабирование домена
        f=rang.shape[0]/domen.shape[0]
        dom_resize=cv2.resize(domen,(0, 0), fx = f, fy = f)
        #Подбираем коэффициенты контраста и яркости
        c,b=self.FindSQ(dom_resize, rang)
        #Преобразование домена
        new_dom=self.ContrastAndBrightness(dom_resize.copy(),c, b)
        #Вычисление ошибки
        err=self.ComputeError2(new_dom, rang)
        # err=self.ComputeError2(dom_resize.copy(), rang)
#        err=self.ComputeError(dom_resize, rang)
        return {'Scale': f, 'Contrast': c, 'Brigth': b, 'Error': err}


if __name__ == '__main__':

    #open image 'd:\\CV\\22\\201395601\\10\\00000XXX\\P6740010_00526534_10_srcimg_0045.tif'
    file = 'd:\\CV\\22\\201395601\\10\\00000XXX\\P6740010_00526534_10_srcimg_0045.tif'
    isIm = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
    c_is = isIm.copy()
    ims = []
    results = []
    for j in range(6):
        start_time = time.time()
    #get working part of the image
        img = get_window(c_is.copy(), t_s=j*128)
        # c_win = img.copy()

    #Building fractal model of the image
        # tmp=FractalModel(img, max_levels=6, min_error=38)
        # tmp=FractalModel(img, max_levels=6, min_error=0.5)
        tmp = FractalModel(img, max_levels=6, min_error=0.5,
                            fileDomens='d:\\CV\\Parsytec\\domens\\domens.png')
        t1 = tmp.di_tmp
        t2 = tmp.allRes
        results.append(t2)

    #Painting rectangle of the image
        coord = []
        img_is = img.copy()
        img1 = img
        for i in t2['rang_id']:
            if len(i) > tmp.max_level-6:
                img1 = cv2.rectangle(
                    img, (t1[i][1], t1[i][0]), (t1[i][3], t1[i][2]), (0, 0, 0), 1)
        print("--- %s seconds ---" % (time.time() - start_time))
        ims.append(img1.copy())
        tt=np.concatenate(ims, axis=1)
    #Painting original and finished image
    fig, axs = plt.subplots(2)
    axs[0].imshow(c_is, cmap='gray')
    axs[1].imshow(tt, cmap='gray')
