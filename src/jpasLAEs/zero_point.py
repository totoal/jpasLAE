import pandas as pd
import numpy as np

from jpasLAEs.utils import w_central, mag_to_flux

tile_dict = {
    'minijpasAEGIS001': 2241,
    'minijpasAEGIS002': 2243,
    'minijpasAEGIS003': 2406,
    'minijpasAEGIS004': 2470,
    'jnep': 2520
}

def Zero_point_error(ref_tile_id_Arr, catname):
    '''
    Calculate the zero point errors for a given set of reference TILE_IDs.

    Loads the zero point magnitudes from a CSV file located in the 'csv' directory.
    For each reference TILE_ID in the given 'ref_tile_id_Arr', the function calculates
    an array with the zero point errors for every filter. The resulting array has shape
    (60, N_src), where N_src is the number of reference TILE_IDs in 'ref_tile_id_Arr'.
    The function returns the resulting array.

    Parameters:
        ref_tile_id_Arr (np.ndarray): Array of reference TILE_IDs.
        catname (str): Name of the catalog file in the 'csv' directory.

    Returns:
        np.ndarray: Array of shape (60, N_src) with the zero point errors of the photometry.

    Raises:
        FileNotFoundError: If the CSV file for 'catname' is not found in the 'csv' directory.
    '''
    # Load Zero Point magnitudes
    try:
        zpt_cat = pd.read_csv(f'csv/{catname}.TileImage.csv', sep=',', header=1)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{catname}.TileImage.csv' not found in the 'csv' directory.")

    # For each reference TILE_ID, we need an array with the ZPT_ERR for every filter
    if catname == 'jnep':
        ref_tileids = np.array([tile_dict['jnep']])
    elif catname == 'minijpas':
        ref_tileids = np.array([tile_dict['minijpasAEGIS001'],
                                tile_dict['minijpasAEGIS002'],
                                tile_dict['minijpasAEGIS003'],
                                tile_dict['minijpasAEGIS004']])
    else:
        raise ValueError(f"Invalid value for 'catname': '{catname}'")
    
    zpt_err_Arr = np.zeros((len(ref_tileids), 60))
    pm_zpt = np.zeros((60, len(ref_tile_id_Arr)))
    for kkk, ref_tid in enumerate(ref_tileids):
        for fil in range(60):
            where = ((zpt_cat['REF_TILE_ID'] == ref_tid)
                     & (zpt_cat['FILTER_ID'] == fil + 1))
            
            zpt_mag = zpt_cat['ZPT'][where]
            zpt_err = zpt_cat['ERRZPT'][where]
            this_zpt_err = (
                mag_to_flux(zpt_mag, w_central[fil])
                - mag_to_flux(zpt_mag + zpt_err, w_central[fil])
            )
            zpt_err_Arr[kkk, fil] = this_zpt_err

        # The array of shape (60, N_src) with the zpt errors of the photometry
        mask = (ref_tile_id_Arr == ref_tid)
        if not np.any(mask):
            continue
        pm_zpt[:, mask] = zpt_err_Arr[kkk].reshape(-1, 1)
    
    return pm_zpt
