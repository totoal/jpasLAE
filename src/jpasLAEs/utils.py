import numpy as np

import csv
import time

from astropy.table import Table
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from scipy.integrate import quad

import pkg_resources


C = 29979245800  # speed of light in cm/s
w_lya = 1215.67 # Angstroms

# Load central wavelength data once
file_path = pkg_resources.resource_filename(__name__, 'data/FILTERs_table.fits')
data_tab = Table.read(file_path, format='fits')
w_central = data_tab['wavelength']


def mag_to_flux(magnitude, wavelength):
    '''
    Convert magnitude to flux.

    Args:
        magnitude (float or array): The magnitude.
        wavelength (float): The wavelength in Angstroms.

    Returns:
        float or array: The flux.
    '''
    return 10**((magnitude + 48.60) / (-2.5)) * C/wavelength**2 * 1e8


def flux_to_mag(flux, wavelength):
    '''
    Convert flux to magnitude.

    Args:
        flux (float or array): The flux.
        wavelength (float): The wavelength in Angstroms.

    Returns:
        float or array: The magnitude.
    '''
    log_arg = np.atleast_1d(flux * wavelength**2/C * 1e-8).astype(float)
    mag = np.empty_like(log_arg)
    mag[log_arg > 0] = -2.5 * np.log10(log_arg[log_arg > 0]) - 48.60
    mag[log_arg < 0] = np.nan
    return mag


def ang_area(dec0, delta_dec, delta_ra):
    '''
    Calculate the angular area of a box on the sky.

    Args:
        dec0 (float): The central declination coordinate.
        delta_dec (float): The aperture of the box in declination.
        delta_ra (float): The aperture of the box in right ascension.

    Returns:
        float: The angular area in square degrees.
    '''
    # Convert degrees to radians and to spherical coordinates (DEC -> azimuth)
    dec0 = np.pi * 0.5 - np.deg2rad(dec0)
    delta_dec = np.deg2rad(delta_dec) / 2
    delta_ra = np.deg2rad(delta_ra) / 2

    # Define the result of the spherical integral in two parts
    a = np.cos(dec0 - delta_dec) - np.cos(dec0 + delta_dec)
    b = 2 * delta_ra
    ang_area = a * b

    # Convert to square degrees
    ang_area = np.rad2deg(ang_area)**2 / np.pi**2
    return ang_area

def load_filter_tags(filepath=None):
    '''
    Load the filter tags from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        list of str: The filter tags.
    '''
    if filepath is None:
        file_path = 'data/minijpas.Filter.csv'
        file_path = pkg_resources.resource_filename(__name__, file_path)

    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        return [line[1] for line in reader]


def central_wavelength():
    '''
    Load the central wavelengths for all filters.

    Returns
    -------
    numpy.ndarray
        Array of central wavelengths.
    '''
    return np.array(w_central)


def nb_fwhm(nb_ind, give_fwhm=True):
    '''
    Returns the FWHM of a filter in tcurves if give_fwhm is True. If it is False, the
    function returns a tuple with (w_central - fwhm/2, w_central + fwhm/2)

    Parameters
    ----------
    nb_ind : int
        Index of the filter to get the FWHM for.
    give_fwhm : bool, optional
        Whether to return the FWHM (True) or the FWHM range (False).
        Default is True.

    Returns
    -------
    float or tuple of float
        FWHM or FWHM range of the specified filter.
    '''
    fwhm = data_tab["width"][nb_ind]
    if give_fwhm:
        return fwhm
    else:
        return w_central - fwhm / 2, w_central + fwhm / 2


def z_volume(z_min, z_max, area):
    '''
    Computes the comoving volume of a redshift range for a given observation area.

    Parameters
    ----------
    z_min : float
        Minimum redshift.
    z_max : float
        Maximum redshift.
    area : float
        Observation area in square degrees.

    Returns
    -------
    volume : astropy.Quantity
        Comoving volume in cubic Mpc / sr.
    '''

    # Define the integrand
    integrand = lambda z: const.c / const.H0 * const.Mpc / (const.kpc * 1e3) \
                        * const.Mpc**2 / const.sr \
                        * cosmo.differential_comoving_volume(z).value

    # Compute the integral
    vol, _ = quad(integrand, z_min, z_max)

    # Compute the solid angle
    area_rad = area * (np.pi / 180)**2
    omega = 2 * np.pi * (1 - np.cos(np.sqrt(area_rad / np.pi)))

    # Convert to astropy.Quantity object
    volume = vol * omega * u.rad**2

    return volume


def z_NB(cont_line_pos):
    '''
    Computes the Lyman-alpha redshift (z) for a given continuum narrowband (NB) index.

    Parameters
    ----------
    cont_line_pos : int or array-like of ints
        Index or indices of the continuum narrowband(s) to compute the redshift for.

    Returns
    -------
    z : float or array-like of floats
        The corresponding redshift(s) of the Lyman-alpha emission line.

    Notes
    -----
    This function assumes that the input continuum narrowband indices correspond to adjacent
    narrowbands centered at wavelengths increasing from the blue to the red end of the spectrum.
    '''
    cont_line_pos = np.atleast_1d(cont_line_pos)

    w1 = w_central[cont_line_pos.astype(int)]
    w2 = w_central[cont_line_pos.astype(int) + 1]

    w = (w2 - w1) * cont_line_pos % 1 + w1

    if len(w) == 1:
        w = w[0]

    return w / 1215.67 - 1


def NB_z(z):
    '''
    Takes a redshift as an argument and returns the index of the corresponding continuum narrowband (NB).
    Returns -1 if the Lyman-alpha redshift is out of the range covered by the narrowbands.

    Parameters
    ----------
    z : float or array-like of floats
        The redshift(s) of the Lyman-alpha emission line(s).

    Returns
    -------
    n_NB : int or array-like of ints
        The index/indices of the corresponding continuum narrowband(s).
    '''
    z = np.atleast_1d(z)

    w_central_NB = w_central[:56]
    w_lya_obs = (z + 1) * 1215.67

    n_NB = np.searchsorted(w_central_NB, w_lya_obs) - 1

    # Set out-of-bounds values to -1
    n_NB[(n_NB < 1) | (n_NB > 54)] = -1

    # If only one value passed, return as a number instead of numpy array
    if len(n_NB) == 1:
        n_NB = n_NB[0]

    return n_NB

def wobs_zlya(w_obs):
    '''
    Calculate the redshift of a Lyman-alpha (Lya) emission line based on the observed wavelength.

    Args:
        w_obs (float): The observed wavelength of the Lya emission line in Angstroms.

    Returns:
        float: The redshift of the Lya emission line.
    '''
    return w_obs / w_lya - 1


def z_volume(z_min, z_max, area):
    '''
    Computes the comoving volume of a redshift range for a given observation area.

    Parameters
    ----------
    z_min : float
        Minimum redshift.
    z_max : float
        Maximum redshift.
    area : float
        Observation area in square degrees.

    Returns
    -------
    volume : astropy.Quantity
        Comoving volume in cubic Mpc.
    '''
    if z_min > z_max:
        raise ValueError('z_min must be less than or equal to z_max.')
    if area <= 0:
        raise ValueError('area must be positive.')

    z_x = np.linspace(z_min, z_max, 1000)
    dV = cosmo.differential_comoving_volume(z_x).to(u.Mpc**3 / u.sr).value
    area_rad = area * (2 * np.pi / 360) ** 2
    theta = np.arccos(1 - area_rad / (2 * np.pi))
    Omega = 2 * np.pi * (1 - np.cos(theta))
    vol = np.trapz(dV, z_x) * Omega
    return vol

def smooth_Image(X_Arr, Y_Arr, Mat, Dx, Dy):
    '''
    X_Arr  es el eje X de la matriz
    Y_Arr  es el eje Y de la matriz
    Mat es la matrix
    Dx  es el delta X que quieres usar para la integracion
    Dx  es el delta Y que quieres usar para la integracion
    '''
    new_Mat = np.zeros_like(Mat)
    for i in range(0, Mat.shape[0]):
        for j in range(0, Mat.shape[1]):
            mask_i = (X_Arr > X_Arr[i] - 0.5 * Dx) * (X_Arr <= X_Arr[i] + 0.5 * Dx)
            mask_j = (Y_Arr > Y_Arr[j] - 0.5 * Dy) * (Y_Arr <= Y_Arr[j] + 0.5 * Dy)

            index_i_Arr = np.arange(0, len(mask_i))
            index_j_Arr = np.arange(0, len(mask_j))

            i_min = np.amin(index_i_Arr[mask_i])
            j_min = np.amin(index_j_Arr[mask_j])
            i_max = np.amax(index_i_Arr[mask_i])
            j_max = np.amax(index_j_Arr[mask_j])

            new_Mat[i, j] = np.sum(Mat[i_min : i_max + 1, j_min : j_max + 1])

    return new_Mat


def bin_centers(bins):
    '''
    Calculates the center values of each bin in a given array of bin edges.

    Args:
        bins (array-like): An array or list containing the bin edges.

    Returns:
        numpy.ndarray: An array containing the center values of each bin.

    Raises:
        ValueError: If the input array `bins` has less than 2 elements.
    '''
    if len(bins) < 2:
        raise ValueError('Input array `bins` must have at least 2 elements.')

    bin_edges = np.asarray(bins).astype(float)
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])

def hms_since_t0(t0, t1=None):
    '''
        Calculates the hours, minutes, and seconds elapsed since a given initial time (t0) until the current time (t1).

    Args:
        t0 (int): The initial time in seconds.
        t1 (int, optional): The end time in seconds. If not provided, the current time will be used.

    Returns:
        tuple: A tuple containing the hours, minutes, and seconds elapsed in that order.
    '''
    t0 = int(t0)
    if t1 is None:
        t1 = int(time.time())

    m, s = divmod(t1 - t0, 60)
    h, m = divmod(m, 60)
    return h, m, s


def smooth_hist(values_Arr, value_min, value_max, step, d_value, weights=None):
    if value_max <= value_min:
        raise ValueError('value_max has to be greater than value_min')

    centers = np.arange(value_min + step * 0.5, value_max, step)
    N_steps = len(centers)
    out_Arr = np.zeros(N_steps, dtype=float)

    for j in range(N_steps):
        this_mask = (
            (values_Arr >= centers[j] - d_value * 0.5)
            & (values_Arr < centers[j] + d_value * 0.5)
        )

        if weights is not None:
            out_Arr[j] = sum(weights[this_mask])
        else:
            out_Arr[j] = sum(this_mask)

    return out_Arr, centers