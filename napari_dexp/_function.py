import napari
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer
from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_experimental_provide_function():
    return [denoise_butterworth_napari]


@register_function(menu="Filtering / noise removal > Butterworth (dexp)")
@time_slicer
def denoise_butterworth_napari(
    image: napari.types.ImageData,
    freq_cutoff: float = 0.5,
    order: float = 1,
    padding: int = 32,
    viewer:napari.Viewer = None) -> napari.types.ImageData:

    from dexp.processing.denoising import denoise_butterworth

    return denoise_butterworth(image,
                               freq_cutoff=freq_cutoff,
                               order=order,
                               padding=padding)

