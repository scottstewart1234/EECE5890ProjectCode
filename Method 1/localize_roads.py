"""
This module plots the coordinates in the map
prerrequisites:

sudo apt install proj-bin libproj-dev libgeos-dev
sudo -H pip3 install https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz
sudo -H pip3 install -U git+https://github.com/matplotlib/basemap.git
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import json


def localize_roads(JSONpath, output_path):
    """
    function to draw coordinades of images in the data base, into a map

    :param JSONpath:
    :param color:
    :param output_path:
    :return:
    """

    # 1. read JSON file
    try:
        with open(JSONpath) as json_file:
            info = json.load(json_file)
    except:
        raise Exception(".json file not found")

    # 2. plot map
    fig = plt.figure(num=None, figsize=(40, 22))
    m = Basemap(width=6000000, height=4500000, resolution='c', projection='aea', lat_1=37, lat_2=38, lon_0=-100,
                lat_0=36)
    m.drawcoastlines(linewidth=0.5)
    m.fillcontinents(color='gray', lake_color='blue')
    m.drawmapboundary(fill_color='royalblue')
    m.drawcountries(linewidth=2, linestyle='solid', color='k')
    m.drawstates(linewidth=1, linestyle='solid', color='k')
    # m.drawmeridians(range(0, 360, 4))

    # 3. plot point on map
    # Region color map
    region_color_map = {
        "Northeast": "blueviolet", "Midwest": "cornflowerblue", "Southeast": "limegreen", "Southwest": "red",
        "West": "orange", -1: "black"
    }

    for point in info:
        lon, lat = m(info[point][0], info[point][1])
        m.scatter(lon, lat, marker='*', color=region_color_map[info[point][3]], zorder=4)

    # 4. save png image
    plt.savefig(output_path)

    return 0


# main
if __name__ == "__main__":
    basefolder = os.path.join("./")
    json_path = os.path.join(basefolder, "index_roads.json")
    output_path = os.path.join(basefolder, "mapPoints.png")
    print(localize_roads(json_path, output_path))
