
import os
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas as gpd
from rasterstats.io import Raster
import glob
import pandas
import geopandas
import numpy as np
import ntpath
import argparse
from tqdm import tqdm
import geopandas


def create_extended_shape_file(input_image, shape_files):
    rasterBand = rasterio.open(input_image)
    outMeta = rasterBand.meta

    if len(shape_files) == 0:
        return outMeta['transform'], outMeta['height'], 0, outMeta['width'], 0, outMeta['crs']

    col_from_min = 100000
    col_to_max = 0
    row_from_min = 100000
    row_to_max = 0
    min_transform = 1000000000
    max_transform = -1
    for i in range(len(shape_files)):

        shape_file = shape_files[i]
        gdf = gpd.read_file(shape_file)
        gdf = gdf[gdf.geometry.notnull()]

        gdf = gdf.to_crs(outMeta['crs'])

        gdf_bounds = gdf.bounds
        gdf = gdf[(gdf_bounds.minx > rasterBand.bounds[0]) & \
                  (gdf_bounds.maxx < rasterBand.bounds[2]) & \
                  (gdf_bounds.miny > rasterBand.bounds[1]) & \
                  (gdf_bounds.maxy < rasterBand.bounds[3])]

        # with fiona.open(shape_file,'r') as file:
        #     aoiGeom = [feature['geometry'] for feature in file if feature['geometry']]
        aoiGeom = gdf['geometry'].tolist()


        _, outTransform2, outWindow = rasterio.mask.raster_geometry_mask(rasterBand, aoiGeom, crop=True)
        min_transform = min(min_transform, outTransform2[2])
        max_transform = max(max_transform, outTransform2[5])

        row_from_min = min(row_from_min, outWindow.row_off)
        col_from_min = min(col_from_min, outWindow.col_off)
        row_to_max = max(row_to_max, outWindow.row_off + outWindow.height)
        col_to_max = max(col_to_max, outWindow.col_off + outWindow.width)

    final_out_transform = rasterio.transform.from_origin(min_transform, max_transform, 10, 10)

    return final_out_transform, row_to_max, row_from_min, col_to_max, col_from_min, outMeta['crs']


def generate_s2(path, geo_info,output_path, create_rgb=False):
    all_files = glob.glob(path + "/" + "*.tif")
    print(all_files)
    print(path)
    outMeta = {"driver": "Gtiff",
                    "height": geo_info[1] - geo_info[2],
                    "width": geo_info[3] - geo_info[4],
                    "transform": geo_info[0],
                   "count": 4,
               "dtype" : rasterio.uint16,
               'crs': geo_info[5]
               }
    # print(outMeta)
    all_cropped = rasterio.open(output_path+'_all_cropped_s2.tiff', 'w',**outMeta)
    # outMeta.update({"count": 4})
    if create_rgb:
        rgb_cropped = rasterio.open(output_path+'_all_cropped_rgb.tiff', 'w',**outMeta)


    for file in tqdm(sorted(all_files)):
        file_name = ntpath.basename(file)
        print(file_name)
        f = file_name.split('.')[0][-2:]

        if f == '8A':
            f = 9
        else:
            f = int(f)
            if f >= 9:
                f += 1
        if f not in [2, 3, 4, 8]:
            continue
        map_f = {2: 1, 3: 2, 4: 3, 8: 4}
        f = map_f[f]
        band = rasterio.open(file)
        band = band.read(1)[geo_info[2]:geo_info[1], geo_info[4]:geo_info[3]]

        all_cropped.write(band,f)


def main(sent_folder_dir, shape_base_path, output_path):

    multipliers = [1, 2, 8, 4, 5]
    # all_folders_names = os.listdir(sent_folder_dir)
    all_folders_names = [sent_folder_dir]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if shape_base_path is not None:
        masks = [shape_base_path]
    else:
        masks = []
    band4_bath = ""
    i = 0
    for name in all_folders_names:
        print('processing')
        print(name)
        i += 1
        input_path = os.path.join(sent_folder_dir, name)
        date = input_path.split('_')[-1]
        band4_bath = glob.glob(os.path.join(input_path, '*B04.jp2'))
        band4_bath = band4_bath[0]
        print(band4_bath)
        geo_info = create_extended_shape_file(band4_bath, masks)
        print(geo_info)
        # print('----geo_info-----')
        # print(geo_info)
        generate_s2(input_path, geo_info, output_path + "_" + str(i))

        print("Done roi generation")




if __name__ == "__main__":

    print("heeey")
    parser = argparse.ArgumentParser()

    parser.add_argument('--sent_folder_dir', type=str,
                        default="/home/developer-2/Desktop/AboHomus_preprocess/",
                        help="path of folders containing sentinel 13 bands")

    parser.add_argument('--shape_base_path', type=str,
                        default="/home/developer-2/Desktop/mask_roi_generation/shape_files_abohomous/",
                        help="path of masks")
    parser.add_argument('--output_path', type=str,
                        default="/home/developer-2/Desktop/phase_1/datasets/input_crop_abohomous/",
                        help="path of output contains the merged lc & roi of sent2")
    parser.add_argument('--data_part', type=str, default=None, help='None or L2A or L1C')
    parser.add_argument('--create_rgb', action='store_true',
                        default=False,
                        help="create rgb.tif")

    args = parser.parse_args()

    main(args.sent_folder_dir, args.shape_base_path, args.output_path, args.data_part, args.create_rgb)


