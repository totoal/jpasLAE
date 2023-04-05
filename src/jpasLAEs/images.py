from astropy.io import fits

import numpy as np

from jpasLAEs.utils import data_tab
fwhm_Arr = data_tab["width"]

# Exposure times for NB and BB in seconds
bb_exp_time = 30
nb_exp_time = 120

default_images_dir = f'/home/alberto/almacen/images_fits'

def load_image(tile_id, filter_id, x_im, y_im, box_side,
               dir=None, normalize=False):
    '''
    Load an image from a FITS file given its tile ID, filter ID, and image coordinates.

    Args:
        tile_id (int): The ID of the tile corresponding to the image.
        filter_id (int): The ID of the filter used for the image.
        x_im (int): The x-coordinate of the center of the image.
        y_im (int): The y-coordinate of the center of the image.
        box_side (int): The size of the box to extract around the center of the image.
        dir (str, optional): The path to the directory containing the images. If not specified, uses the default directory.
        normalize (bool, optional): Whether to normalize the image based on exposure time and FWHM. Default is False.

    Returns:
        numpy.ndarray: A 2D array containing the image data.

    Raises:
        TypeError: If x_im, y_im, or box_side are not integers.
        ValueError: If tile_id does not correspond to any known survey or filter_id is not valid.
    '''
    assert isinstance(box_side, int) or box_side.is_integer()
    assert isinstance(x_im, (int, float))
    assert isinstance(y_im, (int, float))
    assert (filter_id >= 0) and (filter_id < 60)

    x_im = np.round(x_im).astype(int)
    y_im = np.round(y_im).astype(int)

    match tile_id:
        case 2520:
            survey_name = 'jnep'
        case 2241 | 2243 | 2406 | 2470:
            survey_name = 'minijpas'
        case _:
            raise ValueError('tile_id does not correspond to any known survey.')

    if dir is None:
        images_dir = default_images_dir
    else:
        images_dir = dir

    image_name = f'{survey_name}/{tile_id}-{filter_id + 1}.fits'
    
    path_to_image = f'{images_dir}/{image_name}'

    y_range = slice(x_im - box_side - 1, x_im + box_side)
    x_range = slice(y_im - box_side - 1, y_im + box_side)
    im = fits.open(path_to_image)[1].data[x_range, y_range]

    if normalize:
        if filter_id < 56:
            exp_time = nb_exp_time
        else:
            exp_time = bb_exp_time

        im = im / fwhm_Arr[filter_id] * exp_time

    return im