import rm_code.crf as crf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal


def plot_normal(x_range, mu=0, sigma=1, plot_individual_peaks=True):
    """Plot the normal distribution function for a given x range."""
    gaussian = norm.pdf(x_range, mu, scale=sigma)
    if(plot_individual_peaks):
        plt.plot(x_range, gaussian, linewidth=1)
    return(gaussian)


def plot_chromatogram(t_R_list, pw_list, phi_list, t_list, settings_1D, plot_individual_peaks=False):
    """
    Plot chromatogram and gradient profile given a list of retention times
    and peak widths and a gradient profile.

    Args:
        t_R_list: List of retention times.
        pw_list: List of peak widths
        phi_list: List of gradient profile values.
        t_list: List of gradient profile times.
        settings_1D: Dictionary with settings for 1D chromatography:
            t_D: Dead time of the column.
            t_0: Time between the end of the gradient and the end of the column.
            t_init: Time between the start of the gradient and the start of the column.
        plot_individual_peaks: Boolean indicating whether to plot individual peaks.

    Returns:
        None
    """

    t_D, t_0, t_init = settings_1D['t_D'], settings_1D['t_0'], settings_1D['t_init']

    t_R_list, pw_list = crf.sort_peaks(t_R_list, pw_list)
    time_before_gradient_reaches_detector = t_D + t_init + t_0 # +t_0 voor graph van einde kolom
    end_time_profile = t_list[-1] + time_before_gradient_reaches_detector
    end_time_chromatogram = t_R_list[-1] + 2*pw_list[-1]

    # The last peak elutes after the last turning point of the gradient profile
    if(end_time_chromatogram > end_time_profile):
        case = 1
    # The last peak elutes before the last turning point of the gradient profile
    else:
        case = 2

    """Plot chromatogram."""
    fig, ax = plt.subplots()

    # The standard deviation is the pw/4
    sigmas = [pw/4 for pw in pw_list]

    if(case == 1):
        x = np.linspace(0, end_time_chromatogram, 100000)
    else:
        x = np.linspace(0, end_time_profile , 100000)

    #ax.set_xlabel(score)
    sum_of_ys = np.zeros(100000)

    for i, mean in enumerate(t_R_list):
        sigma = sigmas[i]
        y = plot_normal(x, mean, sigma, plot_individual_peaks)
        sum_of_ys = np.add(sum_of_ys, y)

    ax.set_xlim(xmin=0)
    ax.set_xlabel('t (min)')
    # Plot the chromatogram
    ax.plot(x, sum_of_ys, linewidth=1)


    """Plot gradient profile."""
    #phi_list = [phi + 0.01 for phi in phi_list]
    ax2=ax.twinx()
    t_list = [t + time_before_gradient_reaches_detector for t in t_list]

    # Plot dotted red line for time_before_gradient_reaches_detector
    ax2.plot([0, time_before_gradient_reaches_detector], [phi_list[0], phi_list[0]], linestyle="-", linewidth=0.8, color="red", alpha=0.7)

    # Plot the rest of the gradient profile
    ax2.plot(t_list, phi_list, marker=".", markersize=4, linestyle="-", linewidth=0.8, color="black", alpha=0.7)


    if(case == 1):
        ax2.plot([t_list[-1], end_time_chromatogram], [phi_list[-1], phi_list[-1]], marker=".", markersize=4, linestyle="-", linewidth=0.8, color="gray", alpha=0.7)
        ax2.set_xlim(xmin=0, xmax=end_time_chromatogram)
    else:
        ax2.set_xlim(xmin=0, xmax=end_time_profile)

    ax2.set_ylim(ymin=0, ymax=1.1)
    ax2.set_ylabel(r'$\phi$')

    #plt.savefig('fig.png', dpi = 300)
    plt.show()

def plot_shifting_2D_gradient(settings_2D, phi_init, phi_final, t_list):
    """Plots a shifting gradient profile

    Args:
        settings_2D (dict): dictionary containing the settings of the 2D gradient:
            t0 (float): dead time of the 2D gradient
            tI (float): init time of the 2D gradient
            tD (float): dwell time of the 2D gradient
            tG (float): gradient time of the 2D gradient
            tM (float): modulation time of the 2D gradient
        phi_init (list): list of lower bounds of the shifting gradient
        phi_final (list): list of upper bounds of the shifting gradient
        t_list (list): list of time points of the shifting gradient

        Returns:
            fig, ax: matplotlib figure and axis
        """

    t0, tI, tD, tG, tM = settings_2D['t_0'], settings_2D['t_init'], settings_2D['t_D'], settings_2D['t_G'],\
        settings_2D['t_M']
    # initialize figure
    fig,ax = plt.subplots()

    # plot the shifting gradient bounds
    ax.plot(t_list, phi_init, color='tab:blue', label='init')
    ax.plot(t_list, phi_final, color='tab:blue', label='init')

    # set the plot limits
    ax.set_ylim(0, 1)
    ax.set_xlim(0, t_list[-1])

    tau = t0 + tI + tD
    # plot the modulations
    for i in range(int(t_list[-1]//tM)):
        time = i * tM
        ts = [time, time + tau, time + tau + tG, time + tM, time + tM]

        # check between which indices of t_list_2D the time is
        idx = np.searchsorted(t_list, time, side='left')
        # also need to check if the end of the modulation is between the indices
        idx_end = np.searchsorted(t_list, time + tM, side='left')

        # if the modulation ends between the same time segments, we can just use the same slope
        if idx == idx_end:
            phi_init_slope = (phi_init[idx] - phi_init[idx-1]) / (t_list[idx] - t_list[idx-1])
            phi_final_slope = (phi_final[idx] - phi_final[idx-1]) / (t_list[idx] - t_list[idx-1])

            phi_init_init = phi_init_slope * (time - t_list[idx-1]) + phi_init[idx-1]
            phi_init_inter = phi_init_slope * (time + tau - t_list[idx-1]) + phi_init[idx-1]

            phi_init_final = phi_init_slope * (time + tM - t_list[idx-1]) + phi_init[idx-1]

            phi_final_init = phi_final_slope * (time + tG + tau - t_list[idx-1]) + phi_final[idx-1]
            phi_final_inter = phi_final_slope * (time + tM - t_list[idx-1]) + phi_final[idx-1]

            ax.plot(ts, [phi_init_init, phi_init_inter, phi_final_init, phi_final_inter, phi_init_final], color='tab:orange')

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

            ax.plot(ts, [phi_init_init, phi_init_inter, phi_final_init, phi_final_inter, phi_init_final], color='tab:orange')
    return fig, ax

def plot_contour_spectrum(retention_times_x, retention_times_y, widths_x, widths_y, max_T, show_mean=True):
    """
    Plots contour plot of LCxLC data using retention times and base widths
    Essentially creates a xy meshgrid with all zero values in z, then loops over all analytes and computes
    multivariate Gaussians using retention times as mean and (widths/4)**2 as variance.
    show_mean shows mean of gaussians as red dots, to make overlapping analytes more easily visible

    Args:
        retention_times_x (np.array): retention times of first dimension
        retention_times_y (np.array): retention times of second dimension
        widths_x (np.array): widths of first dimension
        widths_y (np.array): widths of second dimension
        max_T (list): list of maximum retention times of both dimensions for plot limits
        show_mean (bool): show mean of gaussians as red dots
    TODO: add functionality that it can support plot limits in a proper way. Current commented lines do not work.
    Returns:
        fig, ax
    """
    # set resolution
    xres = 200
    yres = 200
    # make meshgrid runs from minimum retention time to maximum retention with a correction for average peak
    # widths so that peaks are visible properly
    xs = np.linspace(retention_times_x.min()-np.mean(widths_x), retention_times_x.max()+np.mean(widths_x), xres)
    ys = np.linspace(retention_times_y.max()+np.mean(widths_y), retention_times_y.min()-np.mean(widths_y), yres)
    x, y = np.meshgrid(xs,ys)
    xy = np.column_stack([x.flat, y.flat])
    # inialize z values as zeros
    z = np.zeros(xres*yres)

    # Loop over analytes, extract mean and covariance and add to z.
    for idx, element in enumerate(retention_times_x):
        # if retention_times_x[idx] > max_T[0] or retention_times_y[idx] > max_T[1]:
        #     print('Analyte {} is not plotted because it is outside the maximum retention time.'.format(idx))
        # else:
        mu = np.array([retention_times_x[idx], retention_times_y[idx]])
        covariance = np.diag([(widths_x[idx]/4), (widths_y[idx]/4)])
        z += multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    # Reshape back to a (x, y) grid.
    z3 = z.reshape(x.shape)
    #print(z3.max())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # levels creates the scale of the contour plot. Now runs from 0 to maximum peak intensity in analytes+1 steps.
    c = ax.contourf(x,y,z3, levels=np.linspace(0, z3.max(), len(retention_times_x)+50), cmap='jet')
    fig.colorbar(c)

    # Adds scatter with mean values.

    ax.set_xlabel('Retention time 1D')
    ax.set_ylabel('Retention time 2D')

    # I want to set the limits of the plot to the maximum retention time of each dimension.

    ax.set_xlim(retention_times_x.min()-np.mean(widths_x), retention_times_x.max()+np.mean(widths_x))
    ax.set_ylim(retention_times_y.min()-np.mean(widths_y), retention_times_y.max()+np.mean(widths_y))

    if show_mean==True:
        # skip mean values that are outside the maximum retention time
        for idx, element in enumerate(retention_times_x):
            # if retention_times_x[idx] > max_T[0] or retention_times_y[idx] > max_T[1]:
            #     pass
            #else:
            ax.scatter(retention_times_x[idx], retention_times_y[idx], c='r', s=1)
        #ax.scatter(retention_times_x, retention_times_y, c='r', s=1)

    #plt.xlim(retention_times_x.min()-np.mean(widths_x), retention_times_x.max()+np.mean(widths_x))
    #plt.ylim(retention_times_y.min()-np.mean(widths_y), retention_times_y.max()+np.mean(widths_y))

    plt.show()
    return fig, ax



