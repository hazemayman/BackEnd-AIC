import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage, misc
from scipy.ndimage.filters import gaussian_filter
from PIL import Image

import torch.utils.data as data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
from .data_utils import my_num_classes as num_classes

# mapping from igbp to dfc2020 classes
DFC2020_CLASSES = [
    -1,  # class 0 unused in both schemes
    3,
    3,
    3,
    3,
    3,
    2,
    2,
    2,  # --> will be masked if no_savanna == True
    2,  # --> will be masked if no_savanna == True
    2,
    1,
    1,  # 12 --> 6
    0,  # 13 --> 7
    1,  # 14 --> 6
    6,
    5,
    4
    ]

DFC2020_CLASSES_2 = [
    -1,  # class 0 unused in both schemes
    3,
    2,
    2,
    2,
    1,
    1,
    0,
    6,
    5,
    4
    ]

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR_13 = [2, 3, 4, 8]
S2_BANDS_HR = [1, 2, 3, 4]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [2, 3, 4]  # [1, 10, 11]
RGB_BANDS = [1, 2, 3]


def load_rgb(path):
    image = Image.open(path)
    image.load()
    # new_image = image.resize((256, 256))
    # image.show()
    image_data = np.array(image, np.float32)  # are all you need
    # plt.imshow(image_data)  # Just to verify that image array has been constructed properly
    image_data = image_data.transpose(2, 0, 1)
    # image_data = new_image.astype(np.float32)
    # image_data = np.clip(image_data, 0, 255)
    image_data /= 255
    if np.isnan(image_data).any():
        print("error at: ", path)
    return image_data


def load_rgb_mask(path):
    image_data = load_rgb(path)
    # bands_selected = RGB_BANDS
    # with rasterio.open(path) as data:
    #     s2 = data.read(bands_selected)
    # s2 = s2.astype(np.float32)
    # s2 = s2.astype(np.float32)
    return image_data

# util function for reading s2 data


def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as s2_data:
        if s2_data.meta['count'] > 10:
            bands_selected = S2_BANDS_HR_13
        s2 = s2_data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 /= 10000
    s2 = s2.astype(np.float32)
    if np.isnan(s2).any():
        print("error at: ", path)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as s1_data:
        s1 = s1_data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    if np.isnan(s1).any():
        print("error at: ", path)
    return s1

# util function for reading lc data


def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as lc_data:
        lc = lc_data.read(1)

    lc = lc.astype(np.int64)

    if np.isnan(lc).any():
        print("error at: ", path)
    return lc

# util function for reading lc data
def load_lc_sen(path, no_savanna=False, igbp=True, use_egypt_data=False):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if use_egypt_data:
        lc = lc.astype(np.int64)
    elif igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        # lc = lc.astype(np.int64)
        lc = np.take(DFC2020_CLASSES_2, lc)


    # adjust class scheme to ignore class savanna
    # if no_savanna:
    #     lc[lc == 3] = 0
    #     lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    # lc -= 1
    lc[lc == -1] = 255
    if np.isnan(lc).any():
        print("error at: ", path)
    return lc


def augment_s2_lc(img, lc, ignored_classes=[], ignored_percentage=0.4):
    img = np.rollaxis(img, 0, 3)

    fliplr_p = 0.5
    flipud_p = 0.5
    rotate_p = 0.2
    translate_p = 0.2
    noise_p = -1
    cutout_p = -1
    ignore_p = ignored_percentage

    if np.random.uniform(0, 1) <= ignore_p:
        for ignored_class in ignored_classes:
            lc[lc==ignored_class] = 255
    if np.random.uniform(0, 1) <= fliplr_p:
        img = np.fliplr(img).copy()
        lc = np.fliplr(lc).copy()
    if np.random.uniform(0, 1) <= flipud_p:
        img = np.flipud(img).copy()
        lc = np.flipud(lc).copy()
    if np.random.uniform(0, 1) <= rotate_p:
        ang = np.random.uniform(-45, 45)
        img = ndimage.rotate(img, ang, reshape=False, mode='constant', cval=0)
        lc = ndimage.rotate(lc, ang, reshape=False, mode='constant', cval=255, order=0)
    if np.random.uniform(0, 1) <= cutout_p:
        h, w = img.shape[:2]
        mask_size = int(np.random.uniform(2, 5))
        cxmin, cxmax = 0, w - mask_size
        cymin, cymax = 0, h - mask_size
        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx
        ymin = cy
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        img[ymin:ymax, xmin:xmax, :] = 0
    if np.random.uniform(0, 1) <= noise_p:
        if np.random.uniform(0, 1) <= 0.5:
            img = gaussian_filter(img, sigma=5)
        else:
            blurred_f = gaussian_filter(img, sigma=5)
            filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
            alpha = 2
            img = blurred_f + alpha * (blurred_f - filter_blurred_f)
    if np.random.uniform(0, 1) <= translate_p:
        shift_x = int(np.random.uniform(-25, 25))
        shift_y = int(np.random.uniform(-25, 25))
        new_img = np.zeros_like(img)
        new_lc = np.ones_like(lc) * 255
        if shift_x < 0:
            shift_x = abs(shift_x)
            new_img[:256 - shift_x, :, :] = img[shift_x:, :, :]
            new_lc[:256 - shift_x, :] = lc[shift_x:, :]
        else:
            new_img[shift_x:, :, :] = img[:256 - shift_x, :, :]
            new_lc[shift_x:, :] = lc[:256 - shift_x, :]
        if shift_y < 0:
            shift_y = abs(shift_y)
            new_img[:, :256 - shift_y, :] = img[:, shift_y:, :]
            new_lc[:, :256 - shift_y] = lc[:, shift_y:]
        else:
            new_img[:, shift_y:, :] = img[:, :256 - shift_y, :]
            new_lc[:, shift_y:] = lc[:, :256 - shift_y]
        img = new_img
        lc = new_lc
    img = np.rollaxis(img, 2, 0)
    return img, lc



# util function for reading data from single sample
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, use_rgb=False,
                no_savanna=False, igbp=True, unlabeled=False, augment=False,
                use_egypt_data=False, ignored_classes=[], ignored_percentage=0.4):

    use_s2 = use_s2hr or use_s2mr or use_s2lr
    # print("*******************************************")
    # print("load sample ")
    # print(sample["s2"])
    # print(sample["s1"])
    # print(sample["lc"])

    if use_rgb:
        img = load_rgb(sample["rgb"])
        pass
    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)

    # print("s2 channels: ", img.shape)
    # load s1 data
    if use_s1:
        if use_s2:
            img = np.concatenate((img, load_s1(sample["s1"])), axis=0)
        else:
            img = load_s1(sample["s1"])
        # print("s1 channels: ", img.shape)
    # print("********************************************")
    # load label
    if unlabeled:
        return {'image': img, 'id': sample["id"]}
    else:
        if use_rgb:
            lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
            if augment:
                if np.random.uniform(0, 1) <= 0.5:
                    img = np.fliplr(img).copy()
                    lc = np.fliplr(lc).copy()
                if np.random.uniform(0, 1) <= 0.5:
                    img = np.flipud(img).copy()
                    lc = np.flipud(lc).copy()

        else:
            lc = load_lc_sen(sample["lc"], no_savanna=no_savanna, igbp=igbp, use_egypt_data=use_egypt_data)
            if augment:
                img, lc = augment_s2_lc(img, lc, ignored_classes, ignored_percentage)
            # for ignored_class in [0, 1, 2, 3, 4, 5, 6]:
            #     lc[lc == ignored_class] = 0
            #     # lc[lc == ignored_class*1.0] = 0
            # lc[lc == 7] = 1
    return {'image': img, 'label': lc, 'id': sample["id"]}



# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr, use_rgb):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    if use_rgb:
        n_inputs = 3
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr, use_rgb):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [2, 1, 0]
        brightness_factor = 3
    elif use_rgb:
        display_channels = [0, 1, 2]
        brightness_factor = 1
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)


class SEN12MS(data.Dataset):
    """PyTorch dataset class for the SEN12MS dataset"""
    # expects dataset dir as:
    #       - SEN12MS_holdOutScenes.txt
    #       - ROIsxxxx_y
    #           - lc_n
    #           - s1_n
    #           - s2_n
    #
    # SEN12SEN12MS_holdOutScenes.txt contains the subdirs for the official
    # train/val split and can be obtained from:
    #   https://github.com/MSchmitt1984/SEN12MS/blob/master/splits

    def CreateSentinelSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        s2_locations = []

        for folder in sample_dirs:
            s2_locations += glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
        # pbar = tqdm(total=len(s2_locations))
        # pbar.set_description("[Load]")
        for s2_loc in s2_locations:
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_").replace("_s2", "_s1")
            lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_").replace("_s2", "_lc")
            if id_base_loc:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})
            else:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": s2_loc})
            # pbar.update()
        # pbar.close()
        return sampleFiles

    def CreateRGBSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        rgb_locations = []

        for folder in sample_dirs:
            rgb_locations += glob.glob(os.path.join(path, f"{folder}/*.tiff"), recursive=True)
        #pbar = tqdm(total=len(rgb_locations))
        #pbar.set_description("[Load]")
        for rgb_loc in rgb_locations:
            lc_loc = rgb_loc.replace("rgb", "lc").replace("_rgb", "_lc")
            if id_base_loc:
                sampleFiles.append({"lc": lc_loc, "rgb": rgb_loc, "id": os.path.basename(rgb_loc)})
            else:
                sampleFiles.append({"lc": lc_loc, "rgb": rgb_loc, "id": rgb_loc})
         #   pbar.update()
        #pbar.close()
        return sampleFiles

    def validateSamples(self):
        print("################################")
        print("validate samples")
        newSamples = []
        numberValidSamples = 0
        numberBadSamples = 0
        for x in self.samples:
            lc_loc = x["lc"]
            s1_loc = x["s1"]
            s2_loc = x["s2"]
            with rasterio.open(s2_loc) as s2_dataset:
                #print(s2_loc)
                #print(s2_dataset.profile)
                s2_band_num = s2_dataset.profile["count"]
            with rasterio.open(s1_loc) as s1_dataset:
                # print(s1_loc)
                # print(s1_dataset.profile)
                s1_band_num = s1_dataset.profile["count"]
            with rasterio.open(lc_loc) as lc_dataset:
                # print(lc_loc)
                # print(dataset.profile)
                lc_band_num = lc_dataset.profile["count"]

            if (s2_band_num==13) and (s1_band_num==2) and (lc_band_num==4):
                newSamples.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc,"id": os.path.basename(s2_loc)})
                numberValidSamples = numberValidSamples + 1
            else:
                print("--------------------------------------- Bad Sample -------------------------")
                print(s2_loc)
                print(s2_dataset.profile)
                print(s1_loc)
                print(s1_dataset.profile)
                print(lc_loc)
                print(lc_dataset.profile)
                numberBadSamples = numberBadSamples + 1

        print("numberValidSamples: ", numberValidSamples)
        print("numberBadSamples: ", numberBadSamples)

        print("################################")
        return newSamples

    def __init__(self, path, subset="train", no_savanna=False, use_s2hr=False,
                 use_s2mr=False, use_s2lr=False, use_s1=False, use_rgb=False,
                 unlabeled=False, id_base_loc=True, dataset_type='lc',
                 augment=False, use_egypt_data=False, ignored_classes=[], ignored_percentage=0.4):
        """Initialize the dataset"""

        # inizialize
        super(SEN12MS, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1 or use_rgb):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.use_rgb = use_rgb
        self.no_savanna = no_savanna
        self.unlabeled = unlabeled
        self.id_base_loc = id_base_loc
        self.augment = augment
        self.use_egypt_data = use_egypt_data
        self.ignored_classes = ignored_classes
        self.ignored_percentage = ignored_percentage
        assert subset in ["train", "validation"]

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr,use_rgb)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr, use_rgb)

        # provide number of classes
        self.n_classes = num_classes(dataset_type)
        self.no_savanna = no_savanna

        # if no_savanna:
        #     self.n_classes = max(DFC2020_CLASSES) - 1
        #     self.no_savanna = True
        # else:
        #     self.n_classes = max(DFC2020_CLASSES)
        #     self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # find and index samples
        self.samples = []

        # val_list = list(pd.read_csv(os.path.join(path,
        #                                          "SEN12MS_holdOutScenes.txt"),
        #                             header=None)[0])
        # val_list = [x.replace("s1_", "s2_") for x in val_list]

        # compile a list of paths to all samples
        if (subset == "train") or (subset=="validation"):
            train_list = []
            train_subfolders = []
            if not os.path.isdir(path):
                train_list_base = list(pd.read_csv(path, header=None)[0])
                for seasonfolder in train_list_base:
                    if len(os.path.dirname(seasonfolder)) == 0:
                        seasonfolder = os.path.join(os.path.dirname(path), seasonfolder)
                    train_list += [os.path.join(seasonfolder, x) for x in
                                   os.listdir(seasonfolder)]
            else:
                for file in os.listdir(path):
                    d = os.path.join(path, file)
                    if os.path.isdir(d):
                        train_subfolders.append(file)
                for seasonfolder in train_subfolders:
                    train_list += [os.path.join(seasonfolder, x) for x in
                                   os.listdir(os.path.join(path, seasonfolder))]
                # for seasonfolder in ['ROIs1970_fall', 'ROIs1158_spring',
                #                      'ROIs2017_winter', 'ROIs1868_summer']:
                #     train_list += [os.path.join(seasonfolder, x) for x in
                #                    os.listdir(os.path.join(path, seasonfolder))]
                #
            # print(train_list)
            if not use_rgb: #this is for sentinel data only
                train_list = [x for x in train_list if "s2_" in x]
                # train_list = [x for x in train_list if x not in val_list]
                sample_dirs = train_list
                self.samples = self.CreateSentinelSamplesList(path, sample_dirs, self.id_base_loc)
            else:
                train_list = [x for x in train_list if "rgb" in x]
                self.samples = self.CreateRGBSamplesList(path, train_list, self.id_base_loc)


        # if subset == "train":
        #     pbar = tqdm(total=162556)   # we expect 541,986 / 3 * 0.9 samples
        # else:
        #     pbar = tqdm(total=18106)   # we expect 541,986 / 3 * 0.1 samples
        # pbar.set_description("[Load]")

        # print("Sample training folder: ", sample_dirs)
        #
        # for folder in sample_dirs:
        #     s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"),
        #                              recursive=True)
        #
        #     #print("s2_locations: ")
        #     # for x in s2_locations:
        #     #     print(x)
        #     # print("-----------------------")
        #     # INFO there is one "broken" file in the sen12ms dataset with nan
        #     #      values in the s1 data. we simply ignore this specific sample
        #     #      at this point. id: ROIs1868_summer_xx_146_p202
        #     # if folder == "ROIs1868_summer/s2_146":
        #     #     broken_file = os.path.join(path, "ROIs1868_summer",
        #     #                                "s2_146",
        #     #                                "ROIs1868_summer_s2_146_p202.tif")
        #     #     s2_locations.remove(broken_file)
        #     #     pbar.write("ignored one sample because of nan values in "
        #     #                + "the s1 data")
        #
        #     for s2_loc in s2_locations:
        #         s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
        #         lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")
        #
        #         pbar.update()
        #         self.samples.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc,
        #                              "id": os.path.basename(s2_loc)})
        #
        # pbar.close()
        # self.samples = self.validateSamples()


        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("found", len(self.samples),
              "samples from the sen12ms subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        ret = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_rgb, no_savanna=self.no_savanna,
                           igbp=True, unlabeled=self.unlabeled, augment=self.augment,
                           use_egypt_data=self.use_egypt_data,
                           ignored_classes=self.ignored_classes, ignored_percentage=self.ignored_percentage)
        return ret

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class DFC2020(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 subset="val",
                 no_savanna=False,
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 use_rgb=False,
                 unlabeled=False,
                 id_base_loc=True,
                 dataset_type='lc'):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1 or use_rgb):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.use_rgb = use_rgb
        self.unlabeled = unlabeled
        assert subset in ["val", "test"]
        self.no_savanna = no_savanna
        self.id_base_loc=id_base_loc

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr, use_rgb)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr,use_rgb)

        self.n_classes = num_classes(dataset_type)
        self.no_savanna = no_savanna
        # provide number of classes
        # if no_savanna:
        #     self.n_classes = max(DFC2020_CLASSES) - 1
        # else:
        #     self.n_classes = max(DFC2020_CLASSES)

        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        validation_list = []
        validate_subfolders = []
        for file in os.listdir(path):
            d = os.path.join(path, file)
            if os.path.isdir(d):
                validate_subfolders.append(file)
        for seasonfolder in validate_subfolders:
            validation_list += [os.path.join(seasonfolder, x) for x in
                           os.listdir(os.path.join(path, seasonfolder))]

        if not use_rgb: #this is for sentinel data only
            validation_list = [x for x in validation_list if "s2_" in x]
            sample_dirs = validation_list
            self.samples = self.CreateSentinelSamplesList(path, sample_dirs, self.id_base_loc)
        else:
            validation_list = [x for x in validation_list if "rgb" in x]
            self.samples = self.CreateRGBSamplesList(path, validation_list, self.id_base_loc)

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the dfc2020 subset", subset)


    def CreateSentinelSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        s2_locations = []

        for folder in sample_dirs:
            s2_locations += glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
        #pbar = tqdm(total=len(s2_locations))
        #pbar.set_description("[Load]")
        for s2_loc in s2_locations:
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")
            if id_base_loc:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})
            else:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": s2_loc})
         #   pbar.update()
        #pbar.close()
        return sampleFiles

    def CreateRGBSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        rgb_locations = []

        for folder in sample_dirs:
            rgb_locations += glob.glob(os.path.join(path, f"{folder}/*.tiff"), recursive=True)
        #pbar = tqdm(total=len(rgb_locations))
        #pbar.set_description("[Load]")
        for rgb_loc in rgb_locations:
            lc_loc = rgb_loc.replace("rgb", "lc").replace("_rgb", "_lc")
            if id_base_loc:
                sampleFiles.append({"lc": lc_loc, "rgb": rgb_loc, "id": os.path.basename(rgb_loc)})
            else:
                sampleFiles.append({"lc": lc_loc, "rgb": rgb_loc, "id": rgb_loc})
         #   pbar.update()
        #pbar.close()
        return sampleFiles

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        ret= load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_rgb, no_savanna=self.no_savanna,
                           igbp=False, unlabeled=self.unlabeled)


        return ret

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


class TIFFDir(data.Dataset):
    """PyTorch dataset class for SEN12MS-like data"""
    # expects dataset dir as:
    #       - root
    #           - s1
    #           - s2

    def CreateSentinelSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        s2_locations = []

        for folder in sample_dirs:
            s2_locations += glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
        #pbar = tqdm(total=len(s2_locations))
        #pbar.set_description("[Load]")
        for s2_loc in s2_locations:
            s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
            lc_loc = s2_loc.replace("_s2_", "_lc_").replace("s2_", "lc_")
            if id_base_loc:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": os.path.basename(s2_loc)})
            else:
                sampleFiles.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "id": s2_loc})
         #   pbar.update()
        #pbar.close()
        return sampleFiles

    def CreateRGBSamplesList(self, path, sample_dirs, id_base_loc):
        sampleFiles = []
        rgb_locations = []

        for folder in sample_dirs:
            rgb_locations += glob.glob(os.path.join(path, f"{folder}/*.tiff"), recursive=True)
        #pbar = tqdm(total=len(rgb_locations))
        #pbar.set_description("[Load]")
        for rgb_loc in rgb_locations:
            # lc_loc = rgb_loc.replace("rgb", "lc").replace("_rgb", "_lc")
            if id_base_loc:
                sampleFiles.append({"rgb": rgb_loc, "id": os.path.basename(rgb_loc)})
            else:
                sampleFiles.append({"rgb": rgb_loc, "id": rgb_loc})
        #    pbar.update()
        #pbar.close()
        return sampleFiles

    def __init__(self, path, no_savanna=False, use_s2hr=False,
                 use_s2mr=False, use_s2lr=False, use_s1=False,
                 use_rgb=False, id_base_loc=True, dataset_type='lc'):
        """Initialize the dataset"""

        # inizialize
        super(TIFFDir, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1 or use_rgb):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.use_rgb = use_rgb
        self.no_savanna = no_savanna
        self.id_base_loc = id_base_loc

        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr, use_rgb)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr, use_rgb)

        # provide number of classes
        self.n_classes = num_classes(dataset_type)
        self.no_savanna = no_savanna

        # if no_savanna:
        #     self.n_classes = max(DFC2020_CLASSES) - 1
        #     self.no_savanna = True
        # else:
        #     self.n_classes = max(DFC2020_CLASSES)
        #     self.no_savanna = False

        # make sure parent dir exists
        assert os.path.exists(path)

        # compile a list of paths to all samples
        self.samples = []

        if not use_rgb:
            s2_locations = glob.glob(os.path.join(path, "*/s2_/*.tif*"),
                                 recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                s1_loc = s2_loc.replace("s2/", "s1/").replace("_s2_", "_s1_")
                self.samples.append({"s1": s1_loc, "s2": s2_loc,
                                 "id": os.path.basename(s2_loc)})
        else:
            rgb_locations = glob.glob(os.path.join(path, "rgb/*.tif*"),
                                     recursive=True)
            rgb_locations.extend(glob.glob(os.path.join(path, "rgb/*.tiff"),
                                     recursive=True))
            rgb_locations.extend(glob.glob(os.path.join(path, "rgb/*.jpg"),
                                           recursive=True))
            for rgb_loc in tqdm(rgb_locations, desc="[Load]"):
                self.samples.append({"rgb": rgb_loc,
                                     "id": os.path.basename(rgb_loc)})

        # sort list of samples
        self.samples = sorted(self.samples, key=lambda i: i['id'])

        print("loaded", len(self.samples),
              "samples from the given directory")

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]

        return load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, self.use_rgb , unlabeled=True)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nSEN12MS train")
    # data_dir = "/home/mohamedrehan/research/lukasliebel/train"
    # ds = SEN12MS(data_dir, subset="train", use_s1=False, use_s2hr=True,
    #              use_s2mr=False, no_savanna=True, use_rgb=False)
    data_dir = "/home/mohamedrehan/research/lukasliebel/deepGlobe/train_test"
    ds = SEN12MS(data_dir, subset="train", use_s1=False, use_s2hr=False,
                 use_s2mr=False, no_savanna=True, use_rgb=True)

    s = ds.__getitem__(0)
    print("id:", s["id"], "\n",
          "input shape:", s["image"].shape, "\n",
          "label shape:", s["label"].shape, "\n",
          "number of classes", ds.n_classes)
    train_loader = DataLoader(ds,
                              batch_size=1,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              drop_last=False)
    for batch in tqdm(train_loader):
        pass


# print("\n\nSEN12MS holdout")
    # data_dir = "/root/data"
    # ds = SEN12MS(data_dir, subset="holdout", use_s1=True, use_s2hr=True,
    #              use_s2mr=True, no_savanna=True)
    # s = ds.__getitem__(0)
    # print("id:", s["id"], "\n",
    #       "input shape:", s["image"].shape, "\n",
    #       "label shape:", s["label"].shape, "\n",
    #       "number of classes", ds.n_classes)

    # print("\n\nDFC2020 validation")
    # data_dir = "/root/val"
    # ds = DFC2020(data_dir, subset="val", use_s1=True, use_s2hr=True,
    #              use_s2mr=True, no_savanna=True)
    # s = ds.__getitem__(0)
    # print("id:", s["id"], "\n",
    #       "input shape:", s["image"].shape, "\n",
    #       "label shape:", s["label"].shape, "\n",
    #       "number of classes", ds.n_classes)

    # print("\n\nDFC2020 test")
    # data_dir = "/root/val"
    # ds = DFC2020(data_dir, subset="test", use_s1=True, use_s2hr=True,
    #              use_s2mr=True, no_savanna=True)
    # s = ds.__getitem__(0)
    # print("id:", s["id"], "\n",
    #       "input shape:", s["image"].shape, "\n",
    #       "label shape:", s["label"].shape, "\n",
    #       "number of classes", ds.n_classes)
    #
    # print("\n\nTIFFdir")
    # data_dir = "/root/test"
    # ds = TIFFDir(data_dir, use_s1=True, use_s2hr=True,
    #              use_s2mr=True, no_savanna=True)
    # s = ds.__getitem__(0)
    # print("id:", s["id"], "\n",
    #       "input shape:", s["image"].shape, "\n",
    #       "number of classes", ds.n_classes)(os.path.join(path, "*/s2/*.tif*"))