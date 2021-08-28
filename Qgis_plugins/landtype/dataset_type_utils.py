from enum import Enum
from abc import ABC, abstractmethod


def get_dataset_type_class(dataset_type):
    dataset_type_class = {
        'lc': LandCoverDatasetType,
        'roads': LandCoverRoadsDatasetType,
        'roads_only': RoadsDatasetType,
        'sentinel': SentinelSmallDatasetType,
        'sentinel_egypt': SentinelEgyptDatasetType
    }
    return dataset_type_class[dataset_type]


class DatasetType(ABC):
    def class_names():
        pass

    def class_colors():
        pass

    def class_colors_inv():
        pass

    def number_of_classes():
        pass


class LandCoverDatasetType(DatasetType):

    def class_names():
        return ["Urban-land", "Agriculture_land", "Other green land", "Trees", "Water",
                "Sand-Rocks", "Unknown"]

    def class_colors():
        return ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff',
                '#cc9900', '#000000']

    def class_colors_inv():
        return ['#00ffff', '#ffff00', '#ff00ff', '#00ff00', '#0000ff',
                '#ffffff', '#000000']

    def number_of_classes():
        return 7


class RoadsDatasetType(DatasetType):

    def class_names():
        return ["Road", "Non-Road"]

    def class_colors():
        return ['#ffffff', '#000000']

    def class_colors_inv():
        return ['#ffffff', '#000000']

    def number_of_classes():
        return 2


class LandCoverRoadsDatasetType(DatasetType):

    def class_names():
        return ["Urban-land", "Agriculture_land", "Other green land", "Trees", "Water",
                "Sand-Rocks", "Unknown", "Road"]

    def class_colors():
        return ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff',
                '#cc9900', '#000000', '#A9A9A9']

    def class_colors_inv():
        return ['#00ffff', '#ffff00', '#ff00ff', '#00ff00', '#0000ff',
                '#ffffff', '#ff0000', '#000000']

    def number_of_classes():
        return 8

class SentinelEgyptDatasetType(DatasetType):

    def class_names():
        return ["Urban-land", "Agriculture_land", "Aqua", "Trees", "Water",
                "Sand-Rocks", "Unknown", "Road", 'White-no label']

    def class_colors():
        return ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff',
                '#cc9900', '#000000', '#A9A9A9', '#ffffff']

    def class_colors_inv():
        return ['#00ffff', '#ffff00', '#ff00ff', '#00ff00', '#0000ff',
                '#ffffff', '#ff0000', '#000000', '#ffffff']

    def number_of_classes():
        return 8

class SentinelSmallDatasetType(DatasetType):

    def class_names():
        return ["Urban-land", "Agriculture_land", "Aqua", "Trees", "Water",
                "Sand-Rocks", "Unknown", "Road", 'White-no label']

    def class_colors():
        return ['#ff6600', '#66ff33', '#008080', '#006600', '#0000ff',
                '#cc9900', '#000000', '#A9A9A9']

    def class_colors_inv():
        return ['#00ffff', '#ffff00', '#ff00ff', '#00ff00', '#0000ff',
                '#ffffff', '#ff0000', '#000000']

    def number_of_classes():
        return 8


class SentinelDatasetType(DatasetType):

    def class_names():
        return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
                "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water", "unknown"]

    def class_colors():
        return ['#009900',
                '#c6b044',
                '#fbff13',
                '#b6ff05',
                '#27ff87',
                '#c24f44',
                '#a5a5a5',
                '#69fff8',
                '#f9ffa4',
                '#1c0dff',
                '#ffffff']

    def class_colors_inv():
        return ['#009900',
                '#c6b044',
                '#fbff13',
                '#b6ff05',
                '#27ff87',
                '#c24f44',
                '#a5a5a5',
                '#69fff8',
                '#f9ffa4',
                '#1c0dff',
                '#ffffff']

    def number_of_classes():
        return 10