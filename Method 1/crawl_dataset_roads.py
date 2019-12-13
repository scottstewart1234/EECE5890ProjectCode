"""
Module to crawl the folder and collect a dict into a json file.
"""

import os
import json
import reverse_geocoder as rg  # to determine state name.


def addRegion(gps_point):
    """
    Determines the USA region for a GPS point
    """

    region_name_dict = {
        1: "Northeast", 2: "Midwest", 3: "Southeast", 4: "Southwest", 5: "West"
    }

    regions_dict = {
        "Maine": 1, "Massachusetts": 1, "Rhode Island": 1, "Connecticut": 1, "New Hampshire": 1, "Vermont": 1,
        "New York": 1, "Pennsylvania": 1, "New Jersey": 1, "Delaware": 1, "Maryland": 1,
        "Ohio": 2, "Indiana": 2, "Michigan": 2, "Illinois": 2, "Missouri": 2, "Wisconsin": 2, "Minnesota": 2, "Iowa": 2,
        "Kansas": 2, "Nebraska": 2, "South Dakota": 2, "North Dakota": 2,
        "West Virginia": 3, "Virginia": 3, "Kentucky": 3, "Tennessee": 3, "North Carolina": 3, "South Carolina": 3,
        "Georgia": 3, "Alabama": 3, "Mississippi": 3, "Arkansas": 3, "Louisiana": 3, "Florida": 3,
        "Texas": 4, "Oklahoma": 4, "New Mexico": 4, "Arizona": 4,
        "Colorado": 5, "Wyoming": 5, "Montana": 5, "Idaho": 5, "Washington": 5, "Oregon": 5, "Utah": 5, "Nevada": 5,
        "California": 5, "Alaska": 5, "Hawaii": 5, "British Columbia": 5
    }

    fixed_point = (gps_point[1], gps_point[0])
    state = rg.search(fixed_point)[0]["admin1"]
    if state in regions_dict.keys():
        return region_name_dict[regions_dict[state]]
    else:
        return -1


def list_files(path, format=".jpg"):
    """ls path/*.jpg"""

    files_list = []
    for r, d, f in os.walk(path):
        for file in f:
            if format in file:
                files_list.append(file)
    return files_list


def get_points(fileList):
    """given a list of files 'fileList', get GPS coords."""
    GPSlist = []  # GPSlist is a list of lists of 2 elements [lat, long]
    for name in fileList:
        coords = os.path.splitext(name)  # remove extension
        GPSlist.append(coords[0].split("_"))  # split by '_'
    return GPSlist


def crawl_dataset_roads(data_path, json_path):
    """given the path to data folder 'data_path', creates a JSON file"""

    filename = os.path.join(json_path, "index_roads.json")  # output file

    # 1. Create JSON
    try:
        coords_dict = {}  # init python dict, hash map.
        archivos = list_files(data_path)  # ls path/*.jpg
        gpss = get_points(archivos)  # [[long1 lat1], [long2 lat2], .... [longN latN]]
        generic_id = 0
        for imagen in archivos:  # {ID: [lon, lat, image_file, region]}
            current_point = gpss[generic_id]
            region = addRegion(current_point)
            coords_dict[generic_id] = [current_point[0], current_point[1], imagen, region]
            generic_id += 1

            # save JSON
        with open(filename, 'w') as fichero:
            json.dump(coords_dict, fichero, indent=4)
        return 0
    except:
        return -1


if __name__ == '__main__':
    basefolder = os.path.join("./")
    roads_path = os.path.join(basefolder, "roads")
    print(crawl_dataset_roads(roads_path, basefolder))
