import napari
from napari_tools_menu import register_function
from napari_time_slicer import time_slicer
from napari_plugin_engine import napari_hook_implementation


@napari_hook_implementation
def napari_experimental_provide_function():
    return [denoise_butterworth, lucy_richardson_deconvolution, sobel_filter, area_opening,area_closing,
            area_white_top_hat,area_black_top_hat,dehaze,lipschitz_continuity_correction]


@register_function(menu="Filtering / noise removal > Butterworth (dexp)")
@time_slicer
def denoise_butterworth(
    image: napari.types.ImageData,
    freq_cutoff: float = 0.5,
    order: float = 1,
    padding: int = 32
) -> napari.types.ImageData:
    import dexp

    return dexp.processing.denoising.denoise_butterworth(image=image,
                               freq_cutoff=freq_cutoff,
                               order=order,
                               padding=padding)


@register_function(menu="Filtering / deconvolution > Lucy-Richardson (dexp)")
@time_slicer
def lucy_richardson_deconvolution(
    image: napari.types.ImageData,
    psf: napari.types.ImageData,
    num_iterations: int = 10
) -> napari.types.ImageData:
    import dexp

    return dexp.processing.deconvolution.lucy_richardson_deconvolution(
        image=image,
        psf=psf,
        num_iterations=num_iterations
    )

@register_function(menu="Filtering / edge enhancement > Sobel (dexp)")
@time_slicer
def sobel_filter(
    image: napari.types.ImageData,
    exponent: int = 2,
    gamma: float = 1
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.filters.sobel_filter.sobel_filter(
        image=image,
        exponent=exponent,
        gamma=gamma)


@register_function(menu="Filtering > Area opening (dexp)")
@time_slicer
def area_opening(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.morphology.area_opening(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area closing (dexp)")
@time_slicer
def area_closing(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.morphology.area_closing(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area white top hat (dexp)")
@time_slicer
def area_white_top_hat(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.morphology.area_white_top_hat(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Area black top hat (dexp)")
@time_slicer
def area_black_top_hat(
    image: napari.types.ImageData,
    area_threshold: float = 100,
    sampling: int = 1,
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.morphology.area_black_top_hat(image=image, area_threshold=area_threshold, sampling=sampling)


@register_function(menu="Filtering > Dehaze (dexp)")
@time_slicer
def dehaze(
    image: napari.types.ImageData,
    size: int = 21,
    downscale: int = 4,
    minimal_zero_level: float = 0,
    correct_max_level: bool = True
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.restoration.dehazing.dehaze(
        image=image,
        size=size,
        downscale=downscale,
        minimal_zero_level=minimal_zero_level,
        correct_max_level=correct_max_level,
        in_place=False
    )


@register_function(menu="Filtering > Lipschitz continuity correction (dexp)")
@time_slicer
def lipschitz_continuity_correction(
    image: napari.types.ImageData,
    num_iterations: int = 2,
    correction_percentile: float = 0.1,
    lipschitz: float = 0.1,
    max_proportion_corrected: float = 1,
    decimation: int = 8
) -> napari.types.ImageData:

    import dexp

    return dexp.processing.restoration.lipshitz_correction.lipschitz_continuity_correction(
        image=image,
        num_iterations=num_iterations,
        correction_percentile=correction_percentile,
        lipschitz=lipschitz,
        max_proportion_corrected=max_proportion_corrected,
        decimation=decimation,
        in_place = False)

