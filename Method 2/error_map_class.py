#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:48:33 2019

@author: scottstewart
"""

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
    with open('/home/scottstewart/Desktop/Fall 2019/Comp Vis - Ye/error-6.csv', 'r') as f:
        datapoints = list(csv.reader(f, delimiter=','))
    print(min( row[1] for row in datapoints))
    for row in datapoints:
        maximum = max(row[7:])
        for i in range(7,len(row)):
            if(row[i]==maximum):
                row[i] =1
                pass
            else:
                row[i] = 0
                pass
        
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

    lat = []
    lon = []
    alat = []
    alon = []
    color = []
    tDistance = []
    sDistance = []
    tp=0

    for point in dp:
        x, y = m(point[5], point[6])
        lon.append(x)
        lat.append(y)
        if(point[11]==1):
            color.append("blueviolet") #northeast
        elif(point[7]==1):
            color.append("cornflowerblue")#midwest
        elif(point[8]==1):
            color.append("limegreen")#southeast
        elif(point[10]==1):
            color.append("red")#southwest
        elif(point[9]==1):
            color.append("orange")#west
    m.scatter(lon, lat, color =color, marker='*', zorder=4)

    #from sklearn.metrics import r2_score
    error = []
    #print("R2: "+str(r2_score(GT,Pred)))


    # 4. save png image
    plt.savefig("/home/scottstewart/Desktop/map-3.png")
    fig, axs = plt.subplots(1, 1, sharey=False, tight_layout=False)

    # We can set the number of bins with the `bins` kwarg
    axs.hist(tDistance, bins=100)
    #axs[0].hist(tDistance)
  
    
