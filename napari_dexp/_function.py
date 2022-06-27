from typing import Callable
import inspect
from toolz import curry
from functools import wraps


import napari
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer
from napari_plugin_engine import napari_hook_implementation

from dexp.utils.backends import BestBackend, NumpyBackend, dispatch_data_to_backend


@napari_hook_implementation
def napari_experimental_provide_function():
    return [denoise_butterworth, lucy_richardson_deconvolution, sobel_filter, area_opening,area_closing,
            area_white_top_hat, area_black_top_hat, dehaze, lipschitz_continuity_correction]


@curry
def plugin_function(
    function: Callable,
    cupy: bool = True,
) -> Callable:
    # copied and modified from https://github.com/haesleinhuepf/napari-cupy-image-processing/blob/main/napari_cupy_image_processing/_cupy_image_processing.py
    @wraps(function)
    def worker_function(*args, **kwargs):

        sig = inspect.signature(function)
        # create mapping from position and keyword arguments to parameters
        # will raise a TypeError if the provided arguments do not match the signature
        # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
        bound = sig.bind(*args, **kwargs)
        # set default values for missing arguments
        # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments.apply_defaults
        bound.apply_defaults()

        # copy images to GPU, and create output array if necessary
        with BestBackend() if cupy else NumpyBackend() as bkd:
            dispatch_data_to_backend(bkd, [], bound.arguments)  # inplace
            # call the decorated function
            result = bkd.to_numpy(function(*bound.args, **bound.kwargs))

        return result

    worker_function.__module__ = "napari_dexp"

    return worker_function


@register_function(menu="Filtering / noise removal > Butterworth (DEXP)")
@time_slicer
@plugin_function
def denoise_butterworth(
    image: napari.types.ImageData,
    freq_cutoff: float = 0.5,
    order: float = 1,
    padding: int = 32
) -> napari.types.ImageData:
    from dexp.processing import denoising

    return denoising.denoise_butterworth(
        image=image,
        freq_cutoff=freq_cutoff,
        order=order,
        padding=padding,
    )


@register_function(menu="Filtering / deconvolution > Lucy-Richardson (DEXP)")
@time_slicer
@plugin_function
def lucy_richardson_deconvolution(
    image: napari.types.ImageData,
    psf: napari.types.ImageData,
    num_iterations: int = 10
) -> napari.types.ImageData:
    from dexp.processing import deconvolution

    return deconvolution.lucy_richardson_deconvolution(
        image=image,
        psf=psf,
        num_iterations=num_iterations,
    )

@register_function(menu="Filtering / edge enhancement > Sobel (DEXP)")
@time_slicer
@plugin_function
def sobel_filter(
    image: napari.types.ImageData,
    exponent: int = 2,
    gamma: float = 1
) -> napari.types.ImageData:

    from dexp.processing.filters import sobel_filter

    return sobel_filter.sobel_filter(
        image=image,
        exponent=exponent,
        gamma=gamma,
    )


@register_function(menu="Filtering > Area opening (DEXP)")
@time_slicer
@plugin_function(cupy=False)
def area_opening(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    from dexp.processing import morphology

    return morphology.area_opening(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area closing (DEXP)")
@time_slicer
@plugin_function(cupy=False)
def area_closing(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    from dexp.processing import morphology

    return morphology.area_closing(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area white top hat (DEXP)")
@time_slicer
@plugin_function(cupy=False)
def area_white_top_hat(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    from dexp.processing import morphology

    return morphology.area_white_top_hat(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area black top hat (DEXP)")
@time_slicer
@plugin_function(cupy=False)
def area_black_top_hat(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    from dexp.processing import morphology

    return morphology.area_black_top_hat(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Dehaze (DEXP)")
@time_slicer
@plugin_function
def dehaze(
    image: napari.types.ImageData,
    size: int = 21,
    downscale: int = 4,
    minimal_zero_level: float = 0,
    correct_max_level: bool = True
) -> napari.types.ImageData:

    from dexp.processing.restoration import dehazing

    return dehazing.dehaze(
        image=image,
        size=size,
        downscale=downscale,
        minimal_zero_level=minimal_zero_level,
        correct_max_level=correct_max_level,
        in_place=False,
    )


@register_function(menu="Filtering > Lipschitz continuity correction (DEXP)")
@time_slicer
@plugin_function
def lipschitz_continuity_correction(
    image: napari.types.ImageData,
    num_iterations: int = 2,
    correction_percentile: float = 0.1,
    lipschitz: float = 0.1,
    max_proportion_corrected: float = 1,
    decimation: int = 8
) -> napari.types.ImageData:

    from dexp.processing.restoration import lipshitz_correction

    return lipshitz_correction.lipschitz_continuity_correction(
        image=image,
        num_iterations=num_iterations,
        correction_percentile=correction_percentile,
        lipschitz=lipschitz,
        max_proportion_corrected=max_proportion_corrected,
        decimation=decimation,
        in_place = False,
    )

