import numpy as np
import pandas as pd
from jpasLAEs.zero_point import Zero_point_error

def load_minijpas_jnep(cat_dir, cat_list=['minijpas', 'jnep'],
                       flags_mask=True):
    '''
    Load data from CSV files and return arrays of relevant data.

    Args:
        cat_list (list): List of catalog names to load. Default is ['minijpas', 'jnep'].
        selection (bool): If True, returns only the valuable items for visual_inspection.py.
            Default is False.
        flags_mask (bool): If True, drops flagged rows. Default is True.

    Returns:
        If selection is True:
            pm_flx (ndarray): Array of flux measurements.
            pm_err (ndarray): Array of flux errors.
            x_im (ndarray): Array of x-image positions.
            y_im (ndarray): Array of y-image positions.
            tile_id (ndarray): Array of tile IDs.
            number (ndarray): Array of object numbers.
            starprob (ndarray): Array of star probabilities.
            spCl (ndarray): Array of spectral classes.
            photoz (ndarray): Array of photometric redshifts.
            photoz_chi_best (ndarray): Array of chi-squared values for best-fit photo-z.
            photoz_odds (ndarray): Array of photo-z odds.
            RA (ndarray): Array of right ascension coordinates.
            DEC (ndarray): Array of declination coordinates.
        If selection is False:
            pm_flx (ndarray): Array of flux measurements.
            pm_err (ndarray): Array of flux errors.
            tile_id (ndarray): Array of tile IDs.
            pmra_sn (ndarray): Array of signal-to-noise ratios for proper motion in RA.
            pmdec_sn (ndarray): Array of signal-to-noise ratios for proper motion in DEC.
            parallax_sn (ndarray): Array of signal-to-noise ratios for parallax.
            starprob (ndarray): Array of star probabilities.
            starlhood (ndarray): Array of star likelihoods.
            spCl (ndarray): Array of spectral classes.
            zsp (ndarray): Array of spectroscopic redshifts.
            photoz (ndarray): Array of photometric redshifts.
            photoz_chi_best (ndarray): Array of chi-squared values for best-fit photo-z.
            photoz_odds (ndarray): Array of photo-z odds.
            N_minijpas (int): Number of objects in the 'minijpas' catalog.
            x_im (ndarray): Array of x-image positions.
            y_im (ndarray): Array of y-image positions.
            RA (ndarray): Array of right ascension coordinates.
            DEC (ndarray): Array of declination coordinates.
    '''
    # If selection, return the valuable items for visual_inspection.py only
    pm_flx = np.array([]).reshape(60, 0)
    pm_err = np.array([]).reshape(60, 0)
    tile_id = np.array([])
    parallax_sn = np.array([])
    pmra_sn = np.array([])
    pmdec_sn = np.array([])
    starprob = np.array([])
    starlhood = np.array([])
    spCl = np.array([])
    zsp = np.array([])
    photoz = np.array([])
    photoz_odds = np.array([])
    photoz_chi_best = np.array([])
    x_im = np.array([])
    y_im = np.array([])
    RA = np.array([])
    DEC = np.array([])
    number = np.array([])

    N_minijpas = 0
    split_converter = lambda s: np.array(s.split()).astype(float)
    sum_flags = lambda s: np.sum(np.array(s.split()).astype(float))

    for name in cat_list:
        cat = pd.read_csv(f'{cat_dir}/{name}.Flambda_aper3_photoz_gaia_3.csv', sep=',', header=1,
            converters={0: int, 1: int, 2: split_converter, 3: split_converter, 4: sum_flags,
            5: sum_flags})

        cat = cat[np.array([len(x) for x in cat['FLUX_APER_3_0']]) != 0] # Drop bad rows due to bad query

        if flags_mask:
            cat = cat[(cat.FLAGS == 0) & (cat.MASK_FLAGS == 0)] # Drop flagged
        cat = cat.reset_index()

        tile_id_i = cat['TILE_ID'].to_numpy()

        parallax_i = cat['parallax'].to_numpy() / cat['parallax_error'].to_numpy()
        pmra_i = cat['pmra'].to_numpy() / cat['pmra_error'].to_numpy()
        pmdec_i = cat['pmdec'].to_numpy() / cat['pmdec_error'].to_numpy()

        pm_flx_i = np.stack(cat['FLUX_APER_3_0'].to_numpy()).T * 1e-19
        pm_err_i = np.stack(cat['FLUX_RELERR_APER_3_0'].to_numpy()).T * pm_flx_i

        if name == 'minijpas':
            N_minijpas = pm_flx_i.shape[1]
        
        starprob_i = cat['morph_prob_star']
        starlhood_i = cat['morph_lhood_star']

        RA_i = cat['ALPHA_J2000']
        DEC_i = cat['DELTA_J2000']

        pm_err_i = (pm_err_i ** 2 + Zero_point_error(cat['TILE_ID'], name) ** 2) ** 0.5

        spCl_i = cat['spCl']
        zsp_i = cat['zsp']

        photoz_i = cat['PHOTOZ']
        photoz_odds_i = cat['ODDS']
        photoz_chi_best_i = cat['CHI_BEST']

        x_im_i = cat['X_IMAGE']
        y_im_i = cat['Y_IMAGE']

        number_i = cat['NUMBER']

        pm_flx = np.hstack((pm_flx, pm_flx_i))
        pm_err = np.hstack((pm_err, pm_err_i))
        tile_id = np.concatenate((tile_id, tile_id_i))
        pmra_sn = np.concatenate((pmra_sn, pmra_i))
        pmdec_sn = np.concatenate((pmdec_sn, pmdec_i))
        parallax_sn = np.concatenate((parallax_sn, parallax_i))
        starprob = np.concatenate((starprob, starprob_i))
        starlhood = np.concatenate((starlhood, starlhood_i))
        spCl = np.concatenate((spCl, spCl_i))
        zsp = np.concatenate((zsp, zsp_i))
        photoz = np.concatenate((photoz, photoz_i))
        photoz_odds = np.concatenate((photoz_odds, photoz_odds_i))
        photoz_chi_best = np.concatenate((photoz_chi_best, photoz_chi_best_i))
        x_im = np.concatenate((x_im, x_im_i))
        y_im = np.concatenate((y_im, y_im_i))
        RA = np.concatenate((RA, RA_i))
        DEC = np.concatenate((DEC, DEC_i))
        number = np.concatenate((number, number_i))

    cat = {
        'pm_flx': pm_flx,
        'pm_err': pm_err,
        'tile_id': tile_id,
        'number': number,
        'pmra_sn': pmra_sn,
        'pmdec_sn': pmdec_sn,
        'parallax_sn': parallax_sn,
        'starprob': starprob,
        'starlhood': starlhood,
        'spCl': spCl,
        'zsp': zsp,
        'photoz': photoz,
        'photoz_odds': photoz_odds,
        'x_im': x_im,
        'y_im': y_im,
        'RA': RA,
        'DEC': DEC
    }

    return cat