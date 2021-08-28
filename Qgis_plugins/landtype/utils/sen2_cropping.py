from matplotlib import image
from matplotlib import colors
import glob
import numpy as np
import os
from tqdm import tqdm
import imageio
import rasterio
import argparse
from rasterio.plot import reshape_as_image
from skimage import img_as_ubyte
import random

def calculate_classes_area(lc_image,unique_classes=[0,1,2,3,4,5,6,7]):
    '''
    @param lc_image_path: image containing the roi lc
    @return: dic of the area precenatge of each class
    '''
    lc_area = {}
    total_area = lc_image.shape[0] * lc_image.shape[1]
    for c in unique_classes:
        lc_area[c] = (lc_image == c).sum()
    return lc_area


def from_lc_to_mask(input_path):
    colors_ = ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff', '#cc9900', '#000000', '#A9A9A9', '#ffffff']
    cmap = colors.ListedColormap(colors_)
    for file in glob.glob(input_path+'/*.tif*'):
        with rasterio.open(file) as data:
            img_bands = data.read(1).astype(np.uint8)
        img_bands[img_bands == 255] = 8
        img_bands = cmap(img_bands)[:, :, :3]
        image.imsave(file.replace("lc","mask").replace("tif","png"), img_bands,cmap=cmap)


def convert_s2(input_path, output_path, crop_rect, target_size, bands, rgb_mode=False):

    with rasterio.open(input_path) as s2_data:
        s2 = s2_data.read(bands)

    s2 = reshape_as_image(s2)
    cat_image = s2.astype(np.uint16)

    fileName = os.path.splitext(output_path)[0]
    for index in range(len(crop_rect)):
        left = crop_rect[index]["left"]
        top = crop_rect[index]["top"]
        right = left + target_size
        bottom = top + target_size
        new_image = cat_image[top:bottom, left:right]
        new_image = np.moveaxis(new_image, -1, 0)
        new_file_name = fileName + "_" + str(index) + ".tif"
        if(rgb_mode):
            display_channels = [3, 2, 1]
            s2 = new_image.astype(np.float32)
            min_value, max_value = np.min(s2), np.max(s2)
            s2 = np.clip(s2, min_value, max_value)
            s2 /= max_value
            s2 = s2.astype(np.float32)
            s2 = s2[display_channels, :, :]
            rgb = np.rollaxis(s2, 0, 3)
            imageio.imsave(new_file_name.replace('tif', 'png'), img_as_ubyte(rgb))
        else:
            with rasterio.open(new_file_name, 'w',width=target_size, height=target_size,
                               count=len(bands),driver="Gtiff", dtype=rasterio.uint16) as dst:
                dst.write(new_image)

def select_crops(initial_width,initial_height,target_size=255):
    crop_rect = []
    initial_width = initial_width - target_size
    initial_height = initial_height - target_size
    for x in range(0,initial_width,target_size):
        for y in range(0,initial_height,target_size):
            crop_rect.append({"left": x, "top": y})

    print("total number of crops= ", len(crop_rect))
    # print(crop_rect)
    return crop_rect

def is_representative_enough(sample_area,total_area):
    for c in total_area:
        if total_area[c] == 0:
            continue
        ratio = sample_area[c] / total_area[c]
        print(ratio)

        if ratio < .01: #class is 10% of the total image
            return False
    return True

def is_overlapped(top1,left1,top2,left2,target_size):
    # If one rectangle is on left side of other
    if (left2 >= left1+target_size or left1 >= left2+target_size ):
        return False

    # If one rectangle is down the other
    if (top1 >= top2+target_size or top2 >= top1+target_size):
        return False
    return True



def select_crops_random(lc_image,target_size=256,no_samples = 100):
    '''
    @param initial_width: input size
    @param initial_height: input size
    @param target_size: output size
    @return: the coordinates of cropping
    '''

    crop_rect_train = []
    crop_rect_test = {}
    initial_height = lc_image.shape[0]
    initial_width = lc_image.shape[1]
    total_area = calculate_classes_area(lc_image)
    done = False
    while not done:
        start_top_test = random.randint(0, initial_height - target_size)
        start_left_test = random.randint(0, initial_width - target_size)
        start_right_test = start_left_test + target_size
        start_bottom_test = start_top_test + target_size
        sample_test_image = lc_image[start_top_test:start_bottom_test, start_left_test:start_right_test]
        # check that the test image classes have at least 10% of the overall classes
        sample_area = calculate_classes_area(sample_test_image)
        if (is_representative_enough(sample_area,total_area)):
            done = True
            crop_rect_test = {"left": start_left_test, "top": start_top_test}

    # selecting the training folder
    start_top_train = random.randint(0, initial_height - target_size)
    start_left_train = random.randint(0, initial_width - target_size)
    for i in range(no_samples):
        start_top_train = random.randint(0, initial_height - target_size)
        start_left_train = random.randint(0, initial_width - target_size)
        while not is_overlapped(start_top_train, start_left_train,crop_rect_test['top'],crop_rect_test["left"] , target_size):
            start_top_train = random.randint(0, initial_height - target_size)
            start_left_train = random.randint(0, initial_width - target_size)
        crop_rect_train.append({"left": start_left_train, "top": start_top_train})

    return crop_rect_train , crop_rect_test

def get_shape_info(path):
    with rasterio.open(path) as data:
        img_bands = data.read(1)
    return img_bands.shape


def cut_sen2_files(input_folder,lc_file_name, output_base_folder, region, target_size=256):
    filelist = os.listdir(input_folder)
    file_names = np.array([file for file in filelist if file.endswith(('s2.tif','s2.tiff','planet.tif','planet.tiff'))], dtype=object)
    if lc_file_name is None:
        lc_file_name = os.path.join(input_folder, file_names[0])
    initial_height, initial_width = get_shape_info(lc_file_name)

    set_base_folder_name = "set_" + region + "_"
    s2_folder_name = "s2_"
    mask_folder_name = "mask_"
    label_folder_name = "lc_"
    rgb_folder_name = "rgb_"

    with rasterio.open(lc_file_name) as lc_data:
        lc = lc_data.read()

    lc_image = reshape_as_image(lc)

    crop_rect = select_crops(initial_width, initial_height, target_size)
    # folder_number = 300

    #crop_rect, _ = select_crops_random(lc_image)
    folder_number = 0

    num_files = len(file_names)

    for filename in (file_names):
        output_folder_name = output_base_folder + "/" + set_base_folder_name + str(folder_number)
        output_s2_folder_name = output_folder_name + "/" + s2_folder_name
        output_mask_folder_name = output_folder_name + "/" + mask_folder_name
        output_label_folder_name = output_folder_name + "/" + label_folder_name
        output_rgb_folder_name = output_folder_name + "/" + rgb_folder_name

        os.makedirs(output_folder_name, exist_ok=True)
        os.makedirs(output_s2_folder_name, exist_ok=True)
        os.makedirs(output_mask_folder_name, exist_ok=True)
        os.makedirs(output_label_folder_name, exist_ok=True)
        os.makedirs(output_rgb_folder_name, exist_ok=True)

        input_s2_file = os.path.join(input_folder, filename)
        input_lc_file = lc_file_name
        output_base_s2_file = os.path.join(output_s2_folder_name, filename )

        # crop original rgb images
        #s2
        bands = [1, 2, 3, 4]
        convert_s2(input_s2_file, output_base_s2_file,crop_rect,target_size,bands)
        #rgb
        output_base_rgb_file = output_base_s2_file.replace('s2', 'rgb')
        convert_s2(input_s2_file, output_base_rgb_file, crop_rect, target_size, bands,
                   rgb_mode=True)
        #lc
        bands = [1]
        output_base_label_file = output_base_s2_file.replace('s2', 'lc')
        convert_s2(input_lc_file, output_base_label_file, crop_rect, target_size, bands)
        #mask
        from_lc_to_mask(output_label_folder_name)
        folder_number +=1


    print("processed ", num_files, " files")
    print("**************************************")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=str, default="/home/developer-2/Desktop/phase_1/datasets/input_crop_abohomous/",
                        help='path to original data')

    parser.add_argument('--output_base_folder', type=str, default="/home/developer-2/Desktop/phase_1/datasets/output_crop_abohomous/",
                        help='path to output data')
    parser.add_argument('--target_size', type=int, default=256,
                        help='size of output crops')

    parser.add_argument('--region', type=str, default='region',
                        help='name of region')

    parser.add_argument('--lc_file_name', type=str, default='lc_file_name',
                        help='lc_file_name')

    args = parser.parse_args()

    cut_sen2_files(args.input_folder, args.lc_file_name, args.output_base_folder, args.region)


    #calculate_classes_area(lc_image)


