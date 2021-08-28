import os
import numpy as np
import rasterio as rio
from rasterio.mask import mask
import geopandas as gpd
from datetime import datetime , date
import shutil
import re
# from geotools.constants import *


class GeoMask:
    def __init__(self, tile_path, shape_mask_path):
        '''
        params:
        tile_path: path of the $(tilename).tif file
        shape_mask_path: path of the $(filename).shp file
        '''
        self.tile_path = tile_path
        self.shape_mask_path = shape_mask_path

    def mask(self):
        '''
        masks the shape file on the tile
        -----
        out:
        saves the tif file 
        '''
        # extract data from the tile.tif
        raster_band = rio.open(self.tile_path)
        meta_data = raster_band.meta
        raster_data = raster_band.read()

        # extract the shape file data
        gdf = gpd.read_file(self.shape_mask_path)
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf.to_crs(meta_data['crs'])

        # make sure that the shapefile is smaller than the tile
        gdf_bounds = gdf.bounds
        gdf = gdf[(gdf_bounds.minx > raster_band.bounds[0]) &
                  (gdf_bounds.maxx < raster_band.bounds[2]) &
                  (gdf_bounds.miny > raster_band.bounds[1]) &
                  (gdf_bounds.maxy < raster_band.bounds[3])]

        # masking the tile
        goem = gdf['geometry'].tolist()
        outImage, transform, _ = rio.mask.raster_geometry_mask(
            raster_band, goem, crop=False)
        outImage = np.clip(outImage, 0, 1) * 255
        raster_data = np.clip(raster_data + outImage, 0, 255)

        # extracting features
        colors2 = ['#ff6600', '#66ff33', '#008080', '#006600',
                   '#0000ff', '#cc9900', '#000000', '#A9A9A9', '#ffffff']
        dict_colors = {}
        dict_colors[255] = (255, 255, 255, 255)

        # mapping each color to an index-based label
        for i in range(8):
            dict_colors[i] = self._hex_to_rgb(colors2[i])

        # saving file
        a = os.path.join(os.getcwd(),"Data\shapefile_city\Classified_shapes")
        fileName = raster_band.name.split("/")[-1].split(".tif")[0] + "_merged_masked" + ".tif"

        match = re.search(r'\d{4}-\d{2}-\d{2}', self.tile_path)
        Date = datetime.strptime(match.group(), '%Y-%m-%d').date()
        folderIsThere = os.path.isdir(os.path.join(a,str(Date)))
        if(not(folderIsThere)):
            os.mkdir(os.path.join(a , str(Date)))

        
        new_file_name = os.path.join(a,str(Date), fileName)

        with rio.open(new_file_name, 'w', width=outImage.shape[1], height=outImage.shape[0],
                      count=1, driver="Gtiff", dtype=rio.uint16, crs=gdf.crs,
                      transform=transform) as dst:
            dst.nodata = 255
            dst.write(raster_data.astype(np.uint16))
            dst.write_colormap(1, dict_colors)

        # class names according to the label index
        class_names = ["Urban-land", "Agriculture_land", "Aqua", "Trees", "Water",
                       "Sand-Rocks", "Unknown", "Road", 'White-no label']

        # preparing the stats dict
        stats = self.calculate_region_stats(raster_data)
        stats_dict = {}

        for i in range(len(stats)):
            label = class_names[i].lower()
            stats_dict[label] = stats[i]

        return stats_dict , new_file_name ,fileName

    def _hex_to_rgb(self, hex):
        '''
        Converts hexadecimal color representations to RGB based color system
        '''
        hex = hex.lstrip('#')
        hlen = len(hex)
        t = tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
        t = (t[0], t[1], t[2], 255)
        return t

    def calculate_region_stats(self, raster_data, resolution=10):
        '''
        Extracts the statistics of the land
        '''
        raster_data = np.reshape(raster_data, (-1,))
        stats = []
        for i in range(8):
            data_class = raster_data[raster_data == i].shape[0]
            stats.append(data_class*resolution*resolution)
        return stats
