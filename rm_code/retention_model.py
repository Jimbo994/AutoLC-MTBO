import math

import rm_code.peak_width as peak_width
from rm_code.crf import sort_peaks, resolution_score_2D
import numpy as np


def get_k(k_0, S, phi):
    """Return retention factor k for a given phi."""
    k = k_0 * math.exp(-S * phi)
    return(k)


def get_B_list(phi_list, t_list):
    """Return list of slopes; 1 for each gradient segment."""
    B_list = []
    # Loop over all gradient segments except the last one (that is parallel to t-axis).
    for i in range(len(phi_list) - 1):
        # Calculate the slope of each gradient segment.
        B = (phi_list[i + 1] - phi_list[i])/(t_list[i + 1] - t_list[i])
        B_list.append(B)

    return(B_list)


def isocratic_retention_time(t_0, k):
    """Return retention time for a given analyte under isocratic elution."""
    t_R = t_0 * (1 + k)
    return(t_R)

def retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N):
    """
    Return retention time for a given analyte under multisegment gradient elution.
    :k_0: k0 value for the compound.
    :S: S value for the compound.
    :t_0: Column dead time (defined in globals.py).
    :t_D: Dwell time (defined in globals.py).
    :t_init: Length in time of initial isocratic segment.
    :phi_list: List of phi values; one for each turning point in the gradient profile.
    :t_list: List of t values; one for each turning point in the gradient profile.
    :N: Column efficieny (defined in globals.py).
    :return: Retention time and peak width for the compound under the specified
             gradient profile.
    """
    # k_init is the retention factor before the gradient starts.
    phi_init = phi_list[0]
    k_init = get_k(k_0, S, phi_init)

    fractional_migr_before_gradient = (t_D + t_init)/(t_0 * k_init)

    # CASE 1: compound elutes during first isocratic segment.
    if(fractional_migr_before_gradient >= 1):

        # Calculate retention time
        # The analyte is eluted before the gradient reaches it.
        t_R = isocratic_retention_time(t_0, k_init)

        # Calculate peak width
        # Peak width is calculated using isocratic formula
        W = peak_width.get_peak_width_isocratic(N, t_0, k_init)
        return(t_R, W)

    # Otherwise, the analyte is eluted during the gradient program.
    #print('t_list: ', t_list)

    # Get a list of slopes, one for each gradient segment except the final isocratic segment.
    B_list = get_B_list(phi_list, t_list)

    # Loop over slopes to calculate all a_i's.
    as_list = []
    a_ws_list = []
    for i, B in enumerate(B_list):
        phi_i = phi_list[i]
        phi_i_1 = phi_list[i+1]
        t_i = t_list[i]
        t_i_1 = t_list[i+1]

        k_phi_i = get_k(k_0, S, phi_i)
        k_phi_i_1 = get_k(k_0, S, phi_i_1)


        if(B == 0):
            # Retention time addend if B = 0
            a_i = (t_i_1 - t_i)/(t_0 * k_phi_i)
            # Peak width addend if B = 0
            a_i_w = ((t_i_1 - t_i) * (1 + k_phi_i)**2)/(k_phi_i**3)

        else:
            # Retention time addend if B != 0
            a_i = (1/(t_0 * S * B)) * ((1/k_phi_i_1) - (1/k_phi_i))

            # Peak width addend if B != 0
            a_i_w = (1/(S * B)) * ( (1/3)*((1/k_phi_i_1**3) - (1/k_phi_i**3)) +
                                    (1/k_phi_i_1**2 - 1/k_phi_i**2) +
                                    (1/k_phi_i_1 - 1/k_phi_i) )

        as_list.append(a_i)
        a_ws_list.append(a_i_w)

    cum_sum = 0
    cum_sum_w = 0
    elution_before_last_segment = False
    # Loop over a list until the sum reaches 1 and remember the last segment.
    for i, a in enumerate(as_list):
        if((cum_sum + a + fractional_migr_before_gradient) < 1):
            cum_sum = cum_sum + a
            cum_sum_w = cum_sum_w + a_ws_list[i]
        else:
            elution_segment = i
            elution_before_last_segment = True
            # At this point we want to break out of the loop
            break

    # CASE 2 (2 subcases) Compound is eluted during one of the other gradient segments.
    if(elution_before_last_segment == True):
        # Analyte elutes during another gradient segment.
        # Calculate retention time
        t_n = t_list[elution_segment]
        phi_n = phi_list[elution_segment]
        k_phi_n = get_k(k_0, S, phi_n)
        B_n = B_list[elution_segment]

        # Elution segment is isocratic.
        if(B_n == 0):
            # Slope of gradient segment during which analyte elutes is 0
            t_R = t_D + t_init + t_0 + t_n + t_0*k_phi_n * (1 - fractional_migr_before_gradient - cum_sum)

            # Calculate peak width
            cum_sum_w = cum_sum_w + ((t_R - t_0 - t_D - t_init - t_n) * (1 + k_phi_n)**2) / k_phi_n**3
            k_phi_n = get_k(k_0, S, phi_n)
            G = peak_width.get_G(cum_sum_w, k_phi_n, k_init, t_D, t_0, t_init)
            W = peak_width.get_peak_width_gradient(N, t_0, k_phi_n, G)
            return(t_R, W)

        # Elution segment has a gradient.
        else:
            # Slope of gradient segment during which analyte elutes is not 0
            # Calculate retention time
            t_R = t_0 + t_D + t_init + t_n + (1/(S*B_n)) * np.log(1 + t_0*S*B_n*k_phi_n*(1 - fractional_migr_before_gradient - cum_sum))

            # Calculate peak width
            phi_at_elution = phi_n + B_n * (t_R - t_0 - t_D - t_init - t_n)
            k_at_elution = get_k(k_0, S, phi_at_elution)

            cum_sum_w = cum_sum_w + (1/(S * B_n)) * (
                                    (1/3)* ((1/k_at_elution**3) - (1/k_phi_n**3)) +
                                    (1/k_at_elution**2 - 1/k_phi_n**2) +
                                    (1/k_at_elution - 1/k_phi_n)
                                    )

            G = peak_width.get_G(cum_sum_w, k_at_elution, k_init, t_D, t_0, t_init)
            #G = peak_width.poppe(k_0, S, B_n, t_0)
            W = peak_width.get_peak_width_gradient(N, t_0, k_at_elution, G)
            return(t_R, W)

    # CASE 3: Compound is eluted during last isocratic segment.
    else:
        # Analyte elutes during last isocratic segment.
        phi_n = phi_list[-1]
        k_phi_n = get_k(k_0, S, phi_n)
        t_n = t_list[-1]
        t_R = t_D + t_init + t_0 + t_n + t_0*k_phi_n * (1 - fractional_migr_before_gradient - cum_sum)

        # Add last part to peak peak_width
        cum_sum_w = cum_sum_w + ((t_R - t_0 - t_D - t_init - t_n) * (1 + k_phi_n)**2) / (k_phi_n**3)
        G = peak_width.get_G(cum_sum_w, k_phi_n, k_init, t_D, t_0, t_init)
        W = peak_width.get_peak_width_gradient(N, t_0, k_phi_n, G)
        return(t_R, W)

def compute_chromatogram_1D(ret_pars, settings_1D, phi_list, t_list, sorting=False):
    tR_list = []
    W_list = []
    t_0, t_D, t_init, N = settings_1D['t_0'], settings_1D['t_D'], settings_1D['t_init'], settings_1D['N']
    k0_list, S_list = ret_pars['k0_1D'], ret_pars['S_1D']
    # Calculate retention times and peak widths
    for i in range(len(k0_list)):
        k_0 = k0_list[i]
        S = S_list[i]
        tR, W = retention_time_multisegment_gradient(k_0, S, t_0, t_D, t_init, phi_list, t_list, N)
        tR_list.append(tR)
        W_list.append(W)

        # We need to do sorting so in the list are neighbours.
        if sorting:
            tR_list, W_list = sort_peaks(tR_list, W_list)
    return np.array(tR_list), np.array(W_list)


def get_modulation_program(tR, t0, tI, tD, tG, tM, phi_init, phi_final, t_list):
    """
    Returns the time program and phi program of the modulation at the 1D retention time tR

    Args:
        tR : float 1D retention time of analyte
        t0 : float deadtime of the modulation
        tI : float initial time of the modulation
        tD : float dwell time of the modulation
        tG : float gradient time
        tM : float modulation time
        phi_init : array lower bound of shifting gradient
        phi_final : array upper bound of shifting gradient
        t_list : array time program of the 2D shifting program
    """

    # get modulation
    mod = tR // tM

    time = mod * tM

    tau = t0 + tD

    # then times of that modulation will be
    ts = [time, time + tau + tI, time + tG + tau + tI, time + tM, time + tM]

    # now we check on what segment of the shifting program the modulation is.
    # between which indices of t_list_2D is time
    idx = np.searchsorted(t_list, time, side='left')
    # also need to check if the end of the modulation is between the indices
    idx_end = np.searchsorted(t_list, time + tM, side='left')

    if tR >= t_list[-1]:
        #print('1D Retention time is longer than the total time of the 2D program')
        ts_rm = [i - (time + tau) for i in ts][0:-1]
        ts_rm[0] += tau
        ts_rm[-1] += tau
        phis = [phi_init[-1], phi_init[-1], phi_final[-1], phi_final[-1], phi_init[-1]]
        return int(mod), ts, phis, ts_rm, phis[:-1]

    if idx == idx_end and idx == len(t_list):
        print('error')
        print('Retention time', tR)
    # if the modulation ends between the same time segments, we can just use the same slope
    if idx == idx_end:
        phi_init_slope = (phi_init[idx] - phi_init[idx-1]) / (t_list[idx] - t_list[idx-1])
        phi_final_slope = (phi_final[idx] - phi_final[idx-1]) / (t_list[idx] - t_list[idx-1])

        phi_init_init = phi_init_slope * (time - t_list[idx-1]) + phi_init[idx-1]
        phi_init_inter = phi_init_slope * (time + tau - t_list[idx-1]) + phi_init[idx-1]

        phi_init_final = phi_init_slope * (time + tM - t_list[idx-1]) + phi_init[idx-1]

        phi_final_init = phi_final_slope * (time + tG + tau - t_list[idx-1]) + phi_final[idx-1]
        phi_final_inter = phi_final_slope * (time + tM - t_list[idx-1]) + phi_final[idx-1]
        #phi_final_final = phi_final_slope * (time + tM + tau - t_list[idx-1]) + phi_final[idx-1]
        phis = [phi_init_init, phi_init_inter, phi_final_init, phi_final_inter, phi_final_inter]

    # if this is not the case, we need to be a bit more careful and check at each stage in what shifting segment we are
    # we currently assume that one modulation is at max between two segments, so it assumed that
    # the length of three segments altogether can never be shorter than the modulation time
    else:
        phi_init_slope = (phi_init[idx] - phi_init[idx-1]) / (t_list[idx] - t_list[idx-1])
        phi_init_slope_2 = (phi_init[idx_end] - phi_init[idx_end-1]) / (t_list[idx_end] - t_list[idx_end-1])

        phi_final_slope = (phi_final[idx] - phi_final[idx-1]) / (t_list[idx] - t_list[idx-1])
        phi_final_slope_2 = (phi_final[idx_end] - phi_final[idx_end-1]) / (t_list[idx_end] - t_list[idx_end-1])

        # this always uses the first slope
        phi_init_init = phi_init_slope * (time - t_list[idx-1]) + phi_init[idx-1]

        if time + tau < t_list[idx_end-1]:
            phi_init_inter = phi_init_slope * (time + tau - t_list[idx-1]) + phi_init[idx-1]
        else:
            phi_init_inter = phi_init_slope_2 * (time + tau - t_list[idx_end-1]) + phi_init[idx_end-1]

        if time + tM < t_list[idx_end-1]:
            phi_init_final = phi_init_slope * (time + tM - t_list[idx-1]) + phi_init[idx-1]
        else:
            phi_init_final = phi_init_slope_2 * (time + tM - t_list[idx_end-1]) + phi_init[idx_end-1]

        if time + tG + tau < t_list[idx_end-1]:
            phi_final_init = phi_final_slope * (time + tG + tau - t_list[idx-1]) + phi_final[idx-1]
        else:
            phi_final_init = phi_final_slope_2 * (time + tG + tau - t_list[idx_end-1]) + phi_final[idx_end-1]

        if time + tM < t_list[idx_end-1]:
            phi_final_inter = phi_final_slope * (time + tM - t_list[idx-1]) + phi_final[idx-1]
        else:
            phi_final_inter = phi_final_slope_2 * (time + tM - t_list[idx_end-1]) + phi_final[idx_end-1]
        phis = [phi_init_init, phi_init_inter, phi_final_init, phi_final_inter, phi_final_inter]
    # Legacy code of 1D retention modeling requires the ts matrix to not contain the init, dwell and deadtime.
    # So here we remove them from the time list.
    # we leave t_init in there because it might be that this actually not exactly is an isocratic part.
    ts_rm = [i - (time + tau) for i in ts][0:-1]
    ts_rm[0] += tau
    ts_rm[-1] += tau
    return int(mod), ts, phis, ts_rm, phis[:-1]


def compute_chromatogram_2D(ret_pars, settings_2D, tR_list_1D, phi_init_2D,
                            phi_final_2D, t_list_2D):
    """Compute the 2D chromatogram given the 1D retention times and the 2D parameters.

    Args:
        ret_pars (dict): dictionary of retention parameters
        settings_2D (dict): dictionary of 2D settings
        tR_list_1D (list): list of 1D retention times
        phi_init_2D (list): list of 2D initial phi values
        phi_final_2D (list): list of 2D final phi values
        t_list_2D (list): list of 2D time points

    Returns:
        tR_list_2D (list): list of 2D retention times
        W_list_2D (list): list of 2D peak widths
    """

    t0_2D, tD_2D, tG_2D, tI_2D, tM_2D, N_2D = settings_2D['t_0'], settings_2D['t_D'], settings_2D['t_G'], \
        settings_2D['t_init'], settings_2D['t_M'], settings_2D['N']

    k0_list_2D, S_list_2D = ret_pars['k0_2D'], ret_pars['S_2D']
    tR_list_2D = []
    W_list_2D = []

    # loop over each 1D retention time (analyte)
    for tR, k0, S in zip(tR_list_1D, k0_list_2D, S_list_2D):
        # check in which modulation we are and get respective 2D program of that modulation
        # print(tR)
        # check if tR is None, if so, then we just append None to the list and continue
        if np.isnan(tR):
            print(tR, 'is None')
            tR_2D = 30
            W_2D = 1
        else:
            mod, _, _, ts, phis = get_modulation_program(tR, t0_2D, tI_2D, tD_2D, tG_2D, tM_2D, phi_init_2D, phi_final_2D,
                                                     t_list_2D)

            #   now given this 2D gradient and k0 and S we can compute the retention time and peak width
            # we already have the init time in the phis and ts, so we can just set those to 0.
            # this init time might actually not be isocratic due to the shift boundaries, which is why we chose this design.
            tR_2D, W_2D = retention_time_multisegment_gradient(k0, S, t0_2D, tD_2D, 0, phis, ts, N_2D)

        # append to list
        tR_list_2D.append(tR_2D)
        W_list_2D.append(W_2D)
        # print(tR, tR_2D)
    return np.array(tR_list_2D), np.array(W_list_2D)

# create code for online system, which computes retention times and widths based on the retention model and the retention parameters and the resolution score use the compressed variables described above
def online_system(ret_pars, settings_1D, settings_2D, phi_list_1D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D, max_T):
    """
    This function computes the retention times and widths for a given retention model and retention parameters

    Args
    ret_pars: dictionary with retention parameters
    settings_1D: dictionary with 1D retention model settings
    settings_2D: dictionary with 2D retention model settings
    phi_list_1D: list of phi values for 1D gradient
    t_list_1D: list of time values for 1D gradient
    phi_init_2D: list of initial phi values for 2D gradient
    phi_final_2D: list of final phi values for 2D gradient
    t_list_2D: list of time values for 2D gradient
    max_T: list of maximum times for 1D and 2D.

    Returns
    tR_list_1D: list of retention times for 1D gradient
    W_list_1D: list of widths for 1D gradient
    tR_list_2D: list of retention times for 2D gradient
    W_list_2D: list of widths for 2D gradient
    res_score: resolution score
    """
    # compute retention times and widths
    tR_list_1D, W_list_1D = compute_chromatogram_1D(ret_pars, settings_1D, phi_list_1D, t_list_1D)
    tR_list_2D, W_list_2D = compute_chromatogram_2D(ret_pars, settings_2D,  tR_list_1D, phi_init_2D, phi_final_2D, t_list_2D)

    # compute resolution score
    _, _, _, res_score = resolution_score_2D(tR_list_1D, tR_list_2D, W_list_1D,  W_list_2D, max_time=max_T)
    time_score = np.max(tR_list_1D) * 0.1
    return tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score

def offline_system(ret_pars, settings_1D, settings_2D, phi_list_1D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D, max_T, noise, remove_indices):
    """
    This function computes the retention times and widths for a given retention model and retention parameters and adds noise to the retention times and widths. It also removes some peaks from the data using remove_indices.

    Args
    ret_pars: dictionary with retention parameters
    settings_1D: dictionary with 1D retention model settings
    settings_2D: dictionary with 2D retention model settings
    phi_list_1D: list of phi values for 1D gradient
    t_list_1D: list of time values for 1D gradient
    phi_init_2D: list of initial phi values for 2D gradient
    phi_final_2D: list of final phi values for 2D gradient
    t_list_2D: list of time values for 2D gradient
    max_T: list of maximum times for 1D and 2D.
    noise: dictionary with noise parameters
    remove_indices: list of indices of peaks to remove

    Returns
    tR_list_1D: list of retention times for 1D gradient
    W_list_1D: list of widths for 1D gradient
    tR_list_2D: list of retention times for 2D gradient
    W_list_2D: list of widths for 2D gradient
    res_score: resolution score
    """

    # compute retention times and widths
    tR_list_1D, W_list_1D = compute_chromatogram_1D(ret_pars, settings_1D, phi_list_1D, t_list_1D)
    tR_list_2D, W_list_2D = compute_chromatogram_2D(ret_pars, settings_2D,  tR_list_1D, phi_init_2D, phi_final_2D, t_list_2D)

    # add noise to the retention times and widths
    tR_list_1D = tR_list_1D + np.random.normal(0, noise['tR_1D'], len(tR_list_1D))
    tR_list_2D = tR_list_2D + np.random.normal(0, noise['tR_2D'], len(tR_list_2D))
    W_list_1D = W_list_1D + np.abs(np.random.normal(0, noise['W_1D'], len(W_list_1D)))
    W_list_2D = W_list_2D + np.abs(np.random.normal(0, noise['W_2D'], len(W_list_2D)))


    # remove peaks
    tR_list_1D = np.delete(tR_list_1D, remove_indices)
    tR_list_2D = np.delete(tR_list_2D, remove_indices)
    W_list_1D = np.delete(W_list_1D, remove_indices)
    W_list_2D = np.delete(W_list_2D, remove_indices)

    # check if there are any negative values
    # if np.any(tR_list_1D < 0) or np.any(tR_list_2D < 0) or np.any(W_list_1D < 0) or np.any(W_list_2D < 0):
    #     print('Negative values in retention times or widths')
    #     # print in which list the negative values are
    #     print('tR_list_1D: ', np.any(tR_list_1D < 0))
    #     print('tR_list_2D: ', np.any(tR_list_2D < 0))
    #     print('W_list_1D: ', np.any(W_list_1D < 0))
    #     print('W_list_2D: ', np.any(W_list_2D < 0))

    # make values positive
    tR_list_1D = np.abs(tR_list_1D)
    tR_list_2D = np.abs(tR_list_2D)
    W_list_1D = np.abs(W_list_1D)
    W_list_2D = np.abs(W_list_2D)

    # compute resolution score
    _, _, _, res_score = resolution_score_2D(tR_list_1D, tR_list_2D, W_list_1D, W_list_2D, max_time=max_T)
    time_score = np.max(tR_list_1D) * 0.1
    return tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score

def offline_system2(ret_pars, settings_1D, settings_2D, phi_list_1D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D, max_T, noise, remove_indices):
    """
    This function computes the retention times and widths for a given retention model and retention parameters and adds noise to the retention times and widths. It also removes some peaks from the data using remove_indices.

    Args
    ret_pars: dictionary with retention parameters
    settings_1D: dictionary with 1D retention model settings
    settings_2D: dictionary with 2D retention model settings
    phi_list_1D: list of phi values for 1D gradient
    t_list_1D: list of time values for 1D gradient
    phi_init_2D: list of initial phi values for 2D gradient
    phi_final_2D: list of final phi values for 2D gradient
    t_list_2D: list of time values for 2D gradient
    max_T: list of maximum times for 1D and 2D.
    noise: dictionary with noise parameters
    remove_indices: list of indices of peaks to remove

    Returns
    tR_list_1D: list of retention times for 1D gradient
    W_list_1D: list of widths for 1D gradient
    tR_list_2D: list of retention times for 2D gradient
    W_list_2D: list of widths for 2D gradient
    res_score: resolution score
    """

    # compute retention times and widths
    tR_list_1D, W_list_1D = compute_chromatogram_1D(ret_pars, settings_1D, phi_list_1D, t_list_1D)

    # add noise to the retention times and widths of 1D only
    #tR_list_1D = tR_list_1D + np.random.normal(0, noise['tR_1D'], len(tR_list_1D))
    #W_list_1D = W_list_1D + np.abs(np.random.normal(0, noise['W_1D'], len(W_list_1D)))

    # compute retention times and widths for 2D given the noisy 1D retention times
    tR_list_2D, W_list_2D = compute_chromatogram_2D(ret_pars, settings_2D,  tR_list_1D, phi_init_2D, phi_final_2D, t_list_2D)
    #print(len(tR_list_1D), len(tR_list_2D), len(W_list_1D), len(W_list_2D))
    # remove peaks
    tR_list_1D = np.delete(tR_list_1D, remove_indices)
    tR_list_2D = np.delete(tR_list_2D, remove_indices)
    W_list_1D = np.delete(W_list_1D, remove_indices)
    W_list_2D = np.delete(W_list_2D, remove_indices)
    #print(len(tR_list_1D), len(tR_list_2D), len(W_list_1D), len(W_list_2D))

    # check if there are any negative values
    # if np.any(tR_list_1D < 0) or np.any(tR_list_2D < 0) or np.any(W_list_1D < 0) or np.any(W_list_2D < 0):
    #     print('Negative values in retention times or widths')
    #     # print in which list the negative values are
    #     print('tR_list_1D: ', np.any(tR_list_1D < 0))
    #     print('tR_list_2D: ', np.any(tR_list_2D < 0))
    #     print('W_list_1D: ', np.any(W_list_1D < 0))
    #     print('W_list_2D: ', np.any(W_list_2D < 0))

    # make values positive
    tR_list_1D = np.abs(tR_list_1D)
    tR_list_2D = np.abs(tR_list_2D)
    W_list_1D = np.abs(W_list_1D)
    W_list_2D = np.abs(W_list_2D)

    # compute resolution score
    _, _, _, res_score = resolution_score_2D(tR_list_1D, tR_list_2D, W_list_1D, W_list_2D, max_time=max_T)
    time_score = np.max(tR_list_1D) * 0.1
    return tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score