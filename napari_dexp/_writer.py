"""
This module is an example of a barebones writer plugin for napari

It implements the ``napari_get_writer`` and ``napari_write_image`` hook specifications.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs
"""

from napari_plugin_engine import napari_hook_implementation
from typing import Callable, Optional, Any, Dict, List, Tuple
from dexp.datasets.zarr_dataset import ZDataset


SUPPORTED_TYPES = ['image', 'labels']


@napari_hook_implementation
def napari_get_writer(path: str, layer_types: List[str]) -> Optional[Callable]:

    print(1)
    if not (path.endswith('.zarr') or path.endswith('.zarr.zip')):
        print(1.5)
        return None
    print(2)

    print(3)
    for layer_type in layer_types:
        print(3.5, layer_type)
        if layer_type not in SUPPORTED_TYPES:
            return None
    print(4)

    return writer


def writer(path: str, layers_data: List[Tuple[Any, Dict, str]]) -> str:
    dataset = ZDataset(path, mode='w-')

    for data, meta, _ in layers_data:
        dataset.add_channel(meta['name'],
                            shape=data.shape,
                            dtype=data.dtype)
        dataset.write_array(meta['name'], data)

    dataset.close()
    return path


@napari_hook_implementation
def napari_write_image(path: str, data: Any, meta: Dict) -> str:
    return writer(path, [(data, meta, 'image')])


@napari_hook_implementation
def napari_write_labels(path: str, data: Any, meta: Dict) -> str:
    return writer(path, [(data, meta, 'labels')])

