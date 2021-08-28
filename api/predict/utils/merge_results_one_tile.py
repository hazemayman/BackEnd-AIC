from matplotlib import image
from matplotlib import colors
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
from tqdm import tqdm
import imageio
import rasterio
import argparse
from rasterio.plot import reshape_as_image
from skimage import img_as_ubyte
import warnings
from PIL import Image
import re
warnings.filterwarnings("ignore")


def select_crops(initial_width,initial_height,target_size):
    crop_rect = []
    initial_width = initial_width - target_size
    initial_height = initial_height - target_size
    for x in range(0,initial_width,target_size):
        for y in range(0,initial_height,target_size):
            crop_rect.append({"left": x, "top": y})

    print("total number of crops= ", len(crop_rect))
    return crop_rect


def get_shape_info(path):
    with rasterio.open(path) as data:
        img_bands = data.read(1)
    return img_bands.shape

def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    t = tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
    t = (t[0], t[1], t[2], 255)
    return t

def merge_sen2_files(input_folder, lc_file_name, new_folder, target_size=256, gt_id='dfc'):
    filelist = os.listdir(input_folder)
    file_names = np.array([file for file in filelist if file.endswith(('.tif','.tiff'))], dtype=object)
    num_of_images = 0
    num_of_crops = 0
    for file_name in file_names:
        num_of_images = max(num_of_images, int(re.search(r"^_(\d+)", file_name).groups()[0]))
        num_of_crops = max(num_of_crops, int(re.search(r"_(\d+).tif", file_name).groups()[0])+1)
    
    print(num_of_images, num_of_crops)

    initial_height, initial_width = get_shape_info(lc_file_name)
    crop_rect = select_crops(initial_width, initial_height, target_size)

    raster_lc = rasterio.open(lc_file_name)

    colors2 = ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff', '#cc9900', '#000000', '#A9A9A9', '#ffffff']
    cmap = colors.ListedColormap(colors2)
   
    for i in range(num_of_images):
        output = np.ones((1, initial_height, initial_width)) * 255
        for index in range(num_of_crops):
            left = crop_rect[index]["left"]
            top = crop_rect[index]["top"]
            right = left + target_size
            bottom = top + target_size

            index_file_name = os.path.join(input_folder, '_' + str(i+1) + '_all_cropped_' + str(gt_id) + '_' + str(index) + '.tif')
            with rasterio.open(index_file_name) as data:
                img_bands = data.read(1)

            output[0, top:bottom, left:right] = img_bands
        print("new folder is here : new_folder")
        new_file_name = os.path.join(new_folder, 'result__L2A_T36RTV_A020396_20210131T085044_2021-01-31'+ '.tif')

        dict_colors = {}
        for i in range(8):
            dict_colors[i] = hex_to_rgb(colors2[i])
        dict_colors[255] = (255, 255, 255, 255)
        with rasterio.open(new_file_name, 'w',width=initial_width, height=initial_height,
	                               count=1,driver="Gtiff", dtype=rasterio.uint16, crs=raster_lc.meta['crs'],
                                   transform=raster_lc.meta['transform']) as dst:
            dst.nodata = 255
            dst.write(output.astype(np.uint16))
            dst.write_colormap(1, dict_colors)

        output[output==255] = 8
        output = output[0]

        output_bands = cmap(output)[:, :, :3]
        output_bands = output_bands * 255
        output_bands = output_bands.astype(np.uint16)
        # print(np.unique(output_bands))
        # for j in range(3):
        #     new_file_name = os.path.join(new_folder, 'result_' + str(i+1) + '_band_' + str(j)+ '.tif')
        #     with rasterio.open(new_file_name, 'w',width=initial_width, height=initial_height,
        #                             count=1,driver="Gtiff", dtype=rasterio.uint16, crs=raster_lc.meta['crs'],
        #                             transform=raster_lc.meta['transform']) as dst:
        #         # dst.nodata = 255
        #         print(np.unique(output_bands[:, :, j]))
        #         dst.write(output_bands[:, :, j], 1)
        #         # dst.write(output_bands[:, :, 1], 2)
        #         # dst.write(output_bands[:, :, 2], 3)

        # plt.imshow(output, cmap=cmap)
        # plt.waitforbuttonpress(-1)
        # image.imsave(new_file_name.replace('tif', 'png'), output, cmap=cmap)
        print('done---', str(i+1))

    print("**************************************")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=str,\
                        default="/home/developer-1/Desktop/Sentinel/Sentinel/output/sentinel_training_egypt_augment_noAIC_L2A_minia_sharkia5merkez_9_6/results_aic1/",\
                        help='path to original data')

    parser.add_argument('--lc_file_name', type=str,\
                        default="/home/developer-1/Desktop/Data/sentinel_egypt_before_after/aic_data1/before35_l2a/merged_lc.tiff",\
                        help='path to lc shape file')
    parser.add_argument('--new_folder', type=str,\
                        default="/home/developer-1/Desktop/Sentinel/Sentinel/output/sentinel_training_egypt_augment_noAIC_L2A_minia_sharkia5merkez_8_6/one_tile_results_step_aicdata1/",\
                        help='path to new lc merged file')
    parser.add_argument('--target_size', type=int, default=256,
                        help='size of output crops')

    args = parser.parse_args()

    merge_sen2_files(args.input_folder, args.lc_file_name, args.new_folder, target_size=args.target_size)






