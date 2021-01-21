"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .point_cnn import CNNPointDataset
# from .gid_lulc import GidSegmentation
from .building import BuildSegmentation
from .road import RoadSegmentation
from .gid_lulc import GidSegmentation

datasets = {
    'point_cnn': CNNPointDataset,
    'building': BuildSegmentation,
    'road': RoadSegmentation,
    'gid_lulc': GidSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)


def get_cnn_point_dataset(name, **kwargs):
    """cnn_point Datasets"""
    return datasets[name.lower()](**kwargs)