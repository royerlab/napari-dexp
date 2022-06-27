import numpy as np


def test_all_1image_filters():
    from napari_dexp import denoise_butterworth,sobel_filter,area_opening,area_closing,area_white_top_hat,\
        area_black_top_hat,dehaze,lipschitz_continuity_correction

    functions = [denoise_butterworth,sobel_filter,area_opening,area_closing,area_white_top_hat,
                 area_black_top_hat,dehaze,lipschitz_continuity_correction]

    image = np.ones((10,10))
    for function in functions:
        function(image)


def test_deconvolution():
    from napari_dexp import lucy_richardson_deconvolution
    image = np.ones((10,10))
    psf = np.ones((3,3))

    lucy_richardson_deconvolution(image, psf)
