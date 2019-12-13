#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:48:44 2019

@author: scottstewart
"""
clientIDs = ['Ty1KdHA1dnRMb3p0endQdFlVM2NHdzo1NjYyOGUwYTQ3ZGFmMzEx', 'd3Zwa0Viamw3d3g0cnZrbTB1d3BCdzplYjAxYzkyNWRlZjk2OTNi','RnlhOUxLNzFOSmFiNFFtM2ZZUWUyQToyMzM1ZDljN2RhN2U4Zjc0','aVFKX3h4ejlTX3VXS0tGNW9rNlZLZzo2YjhmNzhlOGQ0ZDA1OWVm','Zi15VXFSelZLRDllQUFBcVVZMjF3Zzo3NWI2YWE3NjY5NDNmNDdk'] #CHANGE THIS FOR IT TO WORK
outputPath = './Downloaded/' #MAKE SURE THIS EXISTS ALREADY. IM TOO LAZY TO CREATE IT

from urllib.request import urlopen
from urllib.request import FancyURLopener
import json
import os
import random
import time 
from shapely.geometry import Point, Polygon

start_time = time.time()

#%%
#Download mapillary images for the region specified
def downloadImages(clientID, minLat, maxLat, minLon, maxLon, maxResults, outputPath, imageRes):
    
    #Generate request based on the specified parameters
    url = 'https://a.mapillary.com/v3/images?client_id=' + clientID + '&bbox='+ str(minLon)+','+str(minLat)+',' +str(maxLon)+','+str(maxLat) + '&per_page=' + str(maxResults)
    request = json.loads(urlopen(url).read())['features']
    
    #Enact requests to obtain streetview images
    imageDesc = "thumb-{0}.jpg".format(imageRes)
    downloadList = []
    print(0,'%')e(imageURL, os.path.join(outputPath, filename))
        downloadList.append(
    for i in range(0, len(request)):
        result = request[i]
        #Generate the direct URL
        imageURL = 'https://images.mapillary.com/' + result['properties']['key'] + '/' + imageDesc
        coordinates = "_".join(map(str, result['geometry']['coordinates']))
        
        #Set output filename
        filename = coordinates + '.jpg'

        #Download and note the image
        FancyURLopener().retrieve(imageURL, os.path.join(outputPath, filename))
        downloadList.append([filename])
        print(((i+1)/len(request))*100,'%')
    return downloadList
def pointinpolygon(point1, point2, polygon):

    point3 = Point(point1.coords[0][0],point2.coords[0][1])
    point4 = Point(point2.coords[0][0],point1.coords[0][1])
    if(not point1.within(polygon) or not point2.within(polygon) or not point3.within(polygon) or not point4.within(polygon)):
        return False
    return True

coords = [(48.838533, -124.914772),
          (49.004557, -95.173609),
          (46.029508, -83.505066),
          (42.149465, -82.544377),
          (45.023055, -71.619530),
          (47.391633, -69.068242),
          (44.772017, -66.923036),
          (42.687827, -70.836982),
          (41.699256, -69.854943),
          (41.019553, -71.864035),
          (40.321966, -73.982021),
          (37.242366, -75.781997),
          (35.326609, -75.354425),
          (31.735529, -81.124803),
          (26.604216, -79.968850),
          (25.261999, -80.337653),
          (25.389672, -81.409831),
          (29.941663, -83.942207),
          (29.132653, -90.110785),
          (28.654991, -96.286772),
          (26.878638, -97.473949),
          (25.849824, -97.148458),
          (26.343527, -99.057736),
          (29.756886, -102.077508),
          (28.958463, -103.177921),
          (31.719249, -106.500888),
          (31.700195, -108.148701),
          (31.321498, -108.199506),
          (31.333705, -111.091680),
          (32.490133, -114.800466),
          (32.673279, -115.318075),
          (32.553585, -117.119980),
          (34.530452, -120.587913),
          (40.238672, -124.270882)]
poly = Polygon(coords)
#Specify developer application clientID from mapillary site

#Specify rectangular coordinate box for where to take images from

#Boston
#minLat, maxLat = 42.219128, 42.400914
#minLon, maxLon = -71.191261, -70.858829
maxResults = 1

#Specify maximum image resolution (Options: 320, 640, 1024, 2048)
imageRes = 1024

#Specify location to store downloaded images


#Milwaukee
i = 0
for clientID in clientIDs:
    while(i<10000):
        #generate a 1km by 1km square
        minLat = random.random()*25+25;
        maxLat = minLat+1/(60*1.85)
        minLon = random.random()*59-125
        maxLon = minLon+1/(60*1.85)
        #check for intersects
        point1= Point(minLat,minLon);
        point2= Point(maxLat,maxLon);
        #print(point1.coords[0][0])
        if(pointinpolygon(point1,point2,poly)):
            i+=1;
           
            passed_time = (time.time()-start_time)*1000
            while(passed_time<1):
                 time.sleep(.01);
                 passed_time = (time.time()-start_time)*1000
            start_time = time.time()
            #Download desired images
            downloadList = downloadImages(clientID, minLat, maxLat, minLon, maxLon, maxResults, outputPath, imageRes)
            #print("at" +str(point1.coords[0]))  
        
            #print("not at" +str(point1.coords[0]))  
    
    i = 0

