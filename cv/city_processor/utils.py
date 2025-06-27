import logging
import os
import glob
import math
import shutil
from PIL import Image
import psutil
import random
from scipy import constants


def deg2num(lat_deg, lon_deg, zoom):
    """
    converts lat/lon to pixel coordinates in given zoom of the EPSG:3857 pyramid
    Parameters
    ----------
    lat_deg: float
        latitude in degrees
    lon_deg: float
        longitude in degrees
    zoom: int
        zoom level of the tile
    Returns
    -------
    xtile: int
        xcoodrinate of the tile in xyz system
    ytile: int
        ycoodrinate of the tile in xyz system
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile: float = int((lon_deg + 180.0) / 360.0 * n)
    ytile: float = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    """
    converts pixel coordinates in given zoom of the EPSG:3857 pyramid to lat/lon
    Parameters
    ----------
    xtile: int
        xcoodrinate of the tile in xyz system
    ytile: int
        ycoodrinate of the tile in xyz system
    zoom: int
        zoom level of the tile
    Returns
    -------
    lat_deg: float
    lon_deg: float
    """
    n = 2.0 ** zoom
    lon_deg: float = xtile / n * 360.0 - 180.0
    lat_rad: float = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg: float = math.degrees(lat_rad)
    return lat_deg, lon_deg

def deg2rad(deg):
    return math.radians(deg)

def unnormalize_pixel_coord(coord, imgsz=512):
    return imgsz * coord

def meters_to_feet(meters):
    return meters / constants.foot