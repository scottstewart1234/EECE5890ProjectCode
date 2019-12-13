#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:33:52 2019

@author: scottstewart
"""
import csv 
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import numpy as np
import math

def load_error():
    with open('/home/scottstewart/Desktop/Fall 2019/Comp Vis - Ye/error.csv', 'r') as f:
        datapoints = list(csv.reader(f, delimiter=','))
    print(min( row[1] for row in datapoints))
    for row in datapoints:
        row[0] = (float(row[0])+0.5)*(49.3-25)+26;
        row[1] = (float(row[1])+0.5)*(-124.914+66)-66;
        row[2] = (float(row[2])+0.5)*(49.3-25)+26;
        row[3] = (float(row[3])+0.5)*(-124.914+66)-66;
    return datapoints




# main
if __name__ == "__main__":
    basefolder = os.path.join("./")
    json_path = os.path.join(basefolder, "index_roads.json")
    output_path = os.path.join(basefolder, "mapPoints.png")
    dp = load_error()    
    # 2. plot map
    fig = plt.figure(num=None, figsize=(40, 22))
    m = Basemap(width=6000000, height=4500000, resolution='c', projection='aea', lat_1=37, lat_2=38, lon_0=-100,
                lat_0=36)
    m.latlon = True
    m.drawcoastlines(linewidth=0.5)
    m.fillcontinents(color='gray', lake_color='blue')
    m.drawmapboundary(fill_color='royalblue')
    m.drawcountries(linewidth=2, linestyle='solid', color='k')
    m.drawstates(linewidth=1, linestyle='solid', color='k')
    # m.drawmeridians(range(0, 360, 4))

    # 3. plot point on map
    # Region color map
    GT = []
    Pred = []
    max_distance = 0
    for point in dp:
        GT.append([point[0],point[1]])
        Pred.append([point[2],point[3]])
        distance = math.sqrt(((point[1]-point[3])**2)+((point[0]-point[2])**2))
        if(distance>max_distance):
            max_distance= distance +0.00001
    lat = []
    lon = []
    alat = []
    alon = []
    distance = []
    tDistance = []
    sDistance = []
    tp=0

    for point in dp:
        x, y = m(point[1], point[0])
        alat.append(point[0])
        alon.append(point[1])
        d = math.sqrt((point[1]-point[3])**2+(point[0]-point[2])**2)
        #x,y=m(point[1],point[0])
        lon.append(x)
        lat.append(y)
        sDistance.append(d*d)
        tDistance.append(d)
        if(d*85<200):
                tp+=1
        distance.append(d/max_distance)
    m.scatter(lon, lat, c =distance, marker='*', zorder=4)
    #lat = np.asarray(lat)
    #lon = np.asarray(lon)
    #distance =np.asarray(distance)
    #m.scatter(lon, lat, marker='*', c=distance, zorder=4)
    mean = np.average(alat)
    meanlon = np.average(alon)
    aDistance = []
    for point in dp:
        x, y = m(point[1], point[0])
        
        d = (mean-point[0])**2+(meanlon-point[1])**2
        aDistance.append(d)
    #from sklearn.metrics import r2_score
    error = []
    #print("R2: "+str(r2_score(GT,Pred)))

    print(1-sum(sDistance)/sum(aDistance))
    print(np.std(sDistance/np.average(sDistance)))
    print(len(tDistance))
    print(sum(tDistance)/len(tDistance))
    print(tp/len(tDistance))
    # 4. save png image
    plt.savefig("/home/scottstewart/Desktop/map-3.png")
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=False)

    # We can set the number of bins with the `bins` kwarg
    axs.hist(tDistance, bins=100)
    #axs[0].hist(tDistance)
  
    
