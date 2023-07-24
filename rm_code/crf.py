import numpy as np

import warnings
# to ignore warning where 0 are multiplied with np.inf and become np.nan.
warnings.filterwarnings('ignore')

def resolution(tR1, tR2, W1, W2):
    """Return the resolution of 2 peaks, given tR and W for both peaks."""
    resolution = ((2*abs(tR2-tR1))/(W1+W2))
    return(resolution)


def sort_peaks(retention_times, peak_widths):
    """
    Sort peaks based on retention time
    and return sorted retention time list and peak width list.
    """
    number_of_peaks = len(retention_times)

    # Create a list of tuples, one for each peak (rt, W)
    peak_tuple_list = []
    for i in range(number_of_peaks):
        peak_tuple = (retention_times[i], peak_widths[i])
        peak_tuple_list.append(peak_tuple)
    # Sort according to first element
    peak_tuples_sorted = sorted(peak_tuple_list, key=lambda x: x[0])

    retention_times = []
    peak_widths = []
    for i in range(number_of_peaks):
        retention_times.append(peak_tuples_sorted[i][0])
        peak_widths.append(peak_tuples_sorted[i][1])

    return(retention_times, peak_widths)


def resolution_score_2D(retention_times_x, retention_times_y, widths_x, widths_y, max_res=1.5, max_time=[100,2]):
    """ Computes resolution score according to eq. 8 in http://dx.doi.org/10.1016/j.chroma.2016.04.061
    Uses Equation 12 in Schure 1997 to compute 2D resolution
    widths_x and widths_y are base widths and thus 4*sigma, the code van Deemter code accounts for this
    Made small adjustment to eq. 8.

    we set diagonal values to inf, because we don't want to compare the same analyte when we take the minimum
    resolution between neighbouring analytes.

    Args:
        retention_times_x (np.array): retention times of first dimension
        retention_times_y (np.array): retention times of second dimension
        widths_x (np.array): peak widths of first dimension
        widths_y (np.array): peak widths of second dimension
        max_res (float): maximum resolution between two analytes, default 1.5
        max_time (list): maximum allowed retention time for first and second dimension, default [60,5]
    Returns:
        resolutions (np.array): resolution score between all analytes
        resolutions_time (np.array): resolution score between all analytes, but with late eluting analytes discarded
        min_res (float): sum of all resolutions between closest neighbouring analytes
        min_res_time (float): sum of all resolutions between closest neighbouring analytes punished by time.
    """

    resolutions = np.ones((len(retention_times_x), len(retention_times_y))) * np.inf

    for idx, element in enumerate(retention_times_x):
        for i in range(idx + 1, len(retention_times_x)):
            dx2 = (retention_times_x[i] - retention_times_x[idx])**2
            dy2 = (retention_times_y[i] - retention_times_y[idx])**2
            wx = np.power(2*((widths_x[i]/4) + (widths_x[idx])/4), 2)
            wy = np.power(2*((widths_y[i]/4) + (widths_y[idx])/4), 2)
            res = np.sqrt((dx2/wx) + (dy2/wy))
            resolutions[idx,i] = res
            if res >= max_res:
                resolutions[idx,i] = max_res
                resolutions[i, idx] = max_res
            else:
                resolutions[idx,i] = res
                resolutions[i, idx] = res

    # zero out scores of peaks that have eluted after max_time
    resolutions_time = resolutions * (retention_times_x < max_time[0]) * (retention_times_y < max_time[1])
    resolutions_time = np.nan_to_num(resolutions_time, np.inf)
    return resolutions / max_res, resolutions_time / max_res, np.sum(np.min(resolutions / max_res, axis=0)), \
        np.sum(np.min(resolutions_time / max_res, axis=0))