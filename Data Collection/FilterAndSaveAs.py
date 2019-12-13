#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 10:51:59 2019

@author: scottstewart
"""
import shutil 
import os
from imageio import imread

readfile = "/home/scottstewart/Desktop/Fall 2019/Comp Vis - Ye/image-classification/Downloaded"
savefile = "/media/scottstewart/F4CED965CED92122/PROJECTDATA"

if os.path.exists(savefile):
        shutil.rmtree(savefile)
os.mkdir(savefile)

images = os.listdir(readfile)
i = int(0)
copy = []
while(i<len(images)):
    Iimage = str(images[i]);
    keep = True
    im = imread(readfile +"/"+str(Iimage))
    if(im.shape[1]<=526):
        keep = False
    
#    while(j<len(images) and keep):
#        Jimage = str(images[j]);
#        word1 = Iimage[:-4].split('_')
#        x1 = float(word1[0])
#        y1 = float(word1[1])
#        
#        word = Jimage[:-4].split('_')
#        x2 = float(word[0])
#        y2 = float(word[1])
#        
#        distance= ((x1-x2)**2+(y1-y2)**2)**(1/2)
#        if(distance<0.000001):
#            print(str(i)+" "+ str(j))
#            keep=False
#        j+=1
    if(keep):
        shutil.copyfile(readfile+"/"+str(Iimage), savefile+"/"+str(Iimage))
    i+=1

