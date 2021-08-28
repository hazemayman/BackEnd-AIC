import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches
from .dataset_type_utils import *

import torch


def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def labels_to_dfc(tensor, no_savanna, dataset_type):
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """
    print("datset type", dataset_type)

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class
    if no_savanna:
        for i in range(2, 9):
            out[tensor == i] = i + 1
    else:
        pass

    # transform from zero-based labels to 1-10
    out[tensor != 255] += 1

    # make sure the mask is intact and return transformed labels
    assert np.all((tensor == 255) == (out == 255))
    return out


def display_input_batch(tensor, display_indices=0, brightness_factor=3):

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor, no_savanna=False, datset_type='lc'):

    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    # tensor = labels_to_dfc(tensor, no_savanna)
    tensor = convert_to_np(tensor)

    # colorize labels
    cmap = mycmap(datset_type)
    imgs = []
    for s in range(tensor.shape[0]):
        im = (tensor[s, :, :]) 
        im = cmap(im)[:, :, 0:3]
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def classnames(dataset_type):
    return get_dataset_type_class(dataset_type).class_names()


def mycmap(dataset_type):
    return colors.ListedColormap(get_dataset_type_class(dataset_type).class_colors())


def mycmap_inv(dataset_type):
    return colors.ListedColormap(get_dataset_type_class(dataset_type).class_colors_inv())


def my_num_classes(dataset_type='lc'):
    return get_dataset_type_class(dataset_type).number_of_classes()

#
# Sentinal_information = {
#     "number_of_classes": 10,
#     "class_names":
#         ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
#          "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water", "unknown"],
#     "class_colors":
#         ['#009900',
#          '#c6b044',
#          '#fbff13',
#          '#b6ff05',
#          '#27ff87',
#          '#c24f44',
#          '#a5a5a5',
#          '#69fff8',
#          '#f9ffa4',
#          '#1c0dff',
#          '#ffffff'],
#     "class_colors_inv":
#         ['#009900',
#          '#c6b044',
#          '#fbff13',
#          '#b6ff05',
#          '#27ff87',
#          '#c24f44',
#          '#a5a5a5',
#          '#69fff8',
#          '#f9ffa4',
#          '#1c0dff',
#          '#ffffff']
#                          }
#
# deep_globe_information = {
#     "number_of_classes": 7,
#     "class_names":
#         ["Urban-land", "Agriculture_land", "Rangeland", "Forest_land", "Water",
#         "Barren_land", "Unknown"],
#     "class_colors":
#         ['#ff6600','#66ff33','#ffffff','#006600','#0000ff',
#         '#cc9900','#000000'],
#     "class_colors_inv":
#         ['#00ffff','#ffff00','#ff00ff','#00ff00','#0000ff',
#         '#ffffff','#000000']
#                          }
#
# deep_globe_roads_information = {
#     "number_of_classes": 8,
#     "class_names":
#         ["Urban-land", "Agriculture_land", "Rangeland", "Forest_land", "Water",
#         "Barren_land", "Road","Unknown"],
#     "class_colors":
#         ['#ff6600','#66ff33','#ffffff','#006600','#0000ff',
#         '#cc9900','#A9A9A9','#000000'],
#     "class_colors_inv":
#         ['#00ffff','#ffff00','#ff00ff','#00ff00','#0000ff',
#         '#ffffff','#000000','#ff0000']
#                          }
#
# deep_globe_roads_only_information = {
#     "number_of_classes": 2,
#     "class_names":
#         ["Road", "Non-Road"],
#     "class_colors":
#         ['#ffffff','#000000']
#                          }
#
#
#
# # cat_image[image == 3] = 0  # (Cyan: 011) => Orange
# # cat_image[image == 6] = 1  # (Yellow: 110) => Light Green
# # cat_image[image == 5] = 2  # (Purple: 101) => White
# # cat_image[image == 2] = 3  # (Green: 010) => Dark green
# # cat_image[image == 1] = 4  # (Blue: 001) => same
# # cat_image[image == 7] = 5  # (White: 111)  => brown
# # cat_image[image == 0] = 6  # (Black: 000) => same
#
# # Urban land: 0,255,255 - Man-made, built up areas with human artifacts (can ignore roads for now which is hard to label)
# # Agriculture land: 255,255,0 - Farms, any planned (i.e. regular) plantation, cropland, orchards, vineyards, nurseries, and ornamental horticultural areas; confined feeding operations.
# # Rangeland: 255,0,255 - Any non-forest, non-farm, green land, grass
# # Forest land: 0,255,0 - Any land with x% tree crown density plus clearcuts.
# # Water: 0,0,255 - Rivers, oceans, lakes, wetland, ponds.
# # Barren land: 255,255,255 - Mountain, land, rock, dessert, beach, no vegetation
# # Unknown: 0,0,0 - Clouds and others
#
# def classnames(dataset_type):
#     if dataset_type == 'lc':
#         return deep_globe_information["class_names"]
#     elif dataset_type == 'roads':
#         return deep_globe_roads_information["class_names"]
#     elif dataset_type == 'roads_only':
#         return deep_globe_roads_only_information["class_names"]
#     elif dataset_type == 'sentinel':
#         return Sentinal_information["class_names"]
#
# def mycmap(dataset_type):
#     if dataset_type == 'lc':
#         cmap = colors.ListedColormap(deep_globe_information["class_colors"])
#     elif dataset_type == 'roads':
#         cmap = colors.ListedColormap(deep_globe_roads_information["class_colors"])
#     elif dataset_type == 'roads_only':
#         cmap = colors.ListedColormap(deep_globe_roads_only_information["class_colors"])
#     elif dataset_type == 'sentinel':
#         cmap = colors.ListedColormap(Sentinal_information["class_colors"])
#     return cmap
#
# def mycmap_inv(dataset_type):
#     if dataset_type == 'lc':
#         cmap = colors.ListedColormap(deep_globe_information["class_colors_inv"])
#     elif dataset_type == 'roads':
#         cmap = colors.ListedColormap(deep_globe_roads_information["class_colors_inv"])
#     elif dataset_type == 'roads_only':
#         cmap = colors.ListedColormap(deep_globe_roads_only_information["class_colors_inv"])
#     elif dataset_type == 'sentinel':
#         cmap = colors.ListedColormap(Sentinal_information["class_colors_inv"])
#
#
#     return cmap
#
# def my_num_classes(dataset_type='lc'):
#     if dataset_type == 'lc':
#         return deep_globe_information["number_of_classes"]
#     elif dataset_type == 'roads':
#         return deep_globe_roads_information["number_of_classes"]
#     elif dataset_type == 'roads_only':
#         return deep_globe_roads_only_information["number_of_classes"]
#     elif dataset_type == 'sentinel':
#         return Sentinal_information["number_of_classes"]


def classnames_org():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]


def mycmap_org():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#c24f44',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap


def mypatches(dataset_type):
    patches = []
    for counter, name in enumerate(classnames(dataset_type)):
        patches.append(mpatches.Patch(color=mycmap(dataset_type).colors[counter],
                                      label=name))
    return patches
