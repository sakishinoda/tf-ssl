import math
import argparse

def deg2num(lat_deg, lon_deg, zoom):
    """ Lon./lat. to tile numbers
    From https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

parser = argparse.ArgumentParser(description="Longitude and latitude to tile coordinates")
parser.add_argument('lat_deg', type=float)
parser.add_argument('lon_deg', type=float)
parser.add_argument('zoom', type=int)
args = parser.parse_args()

x, y = deg2num(args.lat_deg, args.lon_deg, args.zoom)
print("https://mt2.google.com/vt/lyrs=s&x={}&y={}&z={}".format(x, y, args.zoom))

