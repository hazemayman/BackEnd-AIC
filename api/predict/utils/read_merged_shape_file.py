import os
import fiona
import rasterio
from rasterio.mask import mask
import argparse
import numpy as np
import glob
import geopandas as gpd

def create_binary_maskfile(input_image,shape_file, multipliers,shape_mask_file,resolution=10):
    final_image = None
    rasterBand = rasterio.open(input_image)
    outMeta = rasterBand.meta
    # print("------shape-----")
    # print(outMeta)

    col_from_min = 100000
    col_to_max = 0
    row_from_min = 100000
    row_to_max = 0
    min_transform = 1000000000
    max_transform = -1

    gdf = gpd.read_file(shape_file)
    gdf = gdf[gdf.geometry.notnull()]

    gdf = gdf.to_crs(outMeta['crs'])
    gdf_bounds = gdf.bounds
    gdf = gdf[(gdf_bounds.minx > rasterBand.bounds[0]) & \
              (gdf_bounds.maxx < rasterBand.bounds[2]) & \
              (gdf_bounds.miny > rasterBand.bounds[1]) & \
              (gdf_bounds.maxy < rasterBand.bounds[3])]

    for i in range(1, 8):
        gdf_class = gdf[(gdf.Class_id == i)|(gdf.Class_id == str(i))]
        if gdf_class.shape[0] == 0:
            continue
        aoiGeom = gdf_class['geometry'].tolist()

        outImage, outTransform = mask(rasterBand, aoiGeom, crop=False)
        outImage2, outTransform2, outWindow = rasterio.mask.raster_geometry_mask(rasterBand, aoiGeom, crop=True)
        min_transform = min(min_transform, outTransform2[2])
        max_transform = max(max_transform, outTransform2[5])

        row_from_min = min(row_from_min, outWindow.row_off)
        col_from_min = min(col_from_min, outWindow.col_off)
        row_to_max = max(row_to_max, outWindow.row_off + outWindow.height)
        col_to_max = max(col_to_max, outWindow.col_off + outWindow.width)

        outImage = np.clip(outImage, 0, 1) * multipliers[i]
        # print(np.unique(outImage))
        if final_image is None:
            final_image = outImage
        else:
            final_image = final_image + outImage

    final_image = final_image[:, row_from_min:row_to_max, col_from_min: col_to_max]

    final_out_transform = rasterio.transform.from_origin(min_transform, max_transform, resolution, resolution )
    outMeta.update({"driver": 'Gtiff',
                    "height": final_image.shape[1],
                    "width": final_image.shape[2],
                    "transform": final_out_transform})

    final_image[final_image == 0] = 256
    final_image -= 1
    outRaster = rasterio.open(shape_mask_file, "w", **outMeta)
    outRaster.write(final_image)
    outRaster.close()
    # print('---------outMeta---------')
    # print(outMeta)

def main(bandPath_path, shape_file_path, output_path, resolution=10):
    multipliers = {
        3: 1,
        1: 2,
        5: 5,
        4: 8,
        2: 4,
        6: 3,
        7: 6
    }
    bandPath_path = os.path.join(bandPath_path, '*B04.tif')
    # print(bandPath_path)
    bandPath_path = glob.glob(bandPath_path)[0]
    # print(bandPath_path)
    create_binary_maskfile(bandPath_path, shape_file_path, multipliers, output_path,
                           resolution=resolution)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--bandPath_path', type=str,
                        default="/home/developer-2/Desktop/phase_1/datasets/Sharqia/Belbais/input_crop/roi_on_shape.tif",
                        help='path to original band (2-3-4-8)')
    parser.add_argument('--shape_file_path', type=str,
                        default='/home/developer-2/Desktop/phase_1/datasets/Sharqia/Belbais/Belbais_F2.shp',
                        help='base path for shape files')
    parser.add_argument('--output_path', type=str,
                        default='/home/developer-2/Desktop/phase_1/datasets/Sharqia/Belbais/input_crop/merged_lc.tif',
                        help='path of output file')

    parser.add_argument('--resolution', type=int,
                        default=10,
                        help='pixel resolution')

    args = parser.parse_args()
    main(args.bandPath_path, args.shape_file_path, args.output_path, args.resolution)

    