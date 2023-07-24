import torch
import random
import os
import numpy as np

import pandas as pd

# Reading the CSV file back into a DataFrame
df_read = pd.read_csv('data/tkwargs.csv')

# Converting DataFrame back to a dictionary
tkwargs = {
    'dtype':eval(df_read['dtype'][0]),
    'device': torch.device(df_read['device'][0])
}

def seed_everything(seed: int):
    # set all random seeds.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def best_so_far(vals):
    shape = vals.shape
    for i in range(shape[0]):
        for j in range(shape[1] - 1):
            if vals[i, j + 1] < vals[i, j]:
                vals[i, j + 1] = vals[i, j]
    return vals


def ci(y, n_trials):
    return 1.96 * y.std(axis=0) / (np.sqrt(n_trials))


def bo_to_rm_2D(pars, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D):
    """
    convenience function to generate friendly format for the retention modeling code.
    The retention modeling code was created seperately from the bo code.

    # Pars vector will look like this: [phi1, phi2, t1, t2, phi_i1, phi_i2, phi_i3, phi_f1, phi_f2, phi_f3, t2_shift, t3_shift]

    Args:
        pars (torch.tensor): the pars vector from the BO code
        fixed_phi_pars_1D (list): list of fixed phi points in the first dimension
        fixed_time_pars_1D (list): list of fixed time points in the first dimension
        fixed_phi_pars_2D (list): list of fixed phi points in the second dimension
        fixed_time_pars_2D (list): list of fixed time points in the second dimension
    Returns:
        phi_list_1D (list): list of phi points in the first dimension for RM code
        t_list_1D (list): list of time points in the first dimension for RM code
        phi_list_2D (list): list of phi points in the second dimension for RM code
        t_list_2D (list): list of time points in the second dimension for RM code
    """

    shape = pars.shape

    t_list_1D = torch.cat((fixed_time_pars_1D[0].repeat(shape[0], 1).to(**tkwargs), pars[:,2:4].to(**tkwargs), fixed_time_pars_1D[1].repeat(shape[0], 1).to(**tkwargs)), axis=1)
    phi_list_1D = torch.cat((fixed_phi_pars_1D[0].repeat(shape[0], 1).to(**tkwargs), pars[:,0:2].to(**tkwargs), fixed_phi_pars_1D[1].repeat(shape[0], 1).to(**tkwargs))
    ,axis=1)

    t_list_2D = torch.cat((fixed_time_pars_2D[0].repeat(shape[0], 1).to(**tkwargs), fixed_time_pars_2D[1].repeat(shape[0], 1).to(**tkwargs), pars[:, 10:12].to(**tkwargs), fixed_time_pars_2D[2].repeat(shape[0], 1).to(**tkwargs)), axis=1)
    phi_list_init = torch.cat((fixed_phi_pars_2D[0].repeat(shape[0], 1).to(**tkwargs), fixed_phi_pars_2D[0].repeat(shape[0], 1).to(**tkwargs), pars[:, 4:7].to(**tkwargs)), axis=1)
    phi_list_final = torch.cat((pars[:, 7:8].to(**tkwargs), pars[:, 7:10].to(**tkwargs), fixed_phi_pars_2D[1].repeat(shape[0], 1).to(**tkwargs)), axis=1)

    # Will need to return the 1D and 2D phi and t lists
    return phi_list_1D.detach().cpu(), t_list_1D.detach().cpu(), phi_list_init.detach().cpu(), phi_list_final.detach().cpu(), t_list_2D.detach().cpu()


def check_pars(settings_2D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D):
    """
    check if the parameters are valid for the 2D retention modeling code.
    Mainly checks for the following:
    - phi_init_2D and phi_final_2D have the same length
    - time and phi program have the same length
    - 2D shifting program should not be longer than the 1D program and should be sorted
    - there is a whole number of t_M_2D in the time program
    - check if the modulation is longer than deadtime+dwelltime+init time + gradient time.
    - check if time between gradient shifts is longer than the modulation time.
    """
    # check if phi_init_2D and phi_final_2D have the same length
    assert len(phi_init_2D) == len(
        phi_final_2D), f"lower and upper bounds of shifting gradient should have the same dimension, got: {len(phi_init_2D), len(phi_final_2D)}"
    # check if time and phi program have the same length
    assert len(phi_init_2D) == len(
        t_list_2D), f"shifting gradient program and time program should have the same dimension, got: {len(phi_init_2D), len(t_list_2D)}"

    # check if 2D shifting program should not be longer than the 1D program and should be sorted
    assert sorted(t_list_2D) == t_list_2D, f"2D time points should be ordered from low to high"
    assert sorted(t_list_1D) == t_list_1D, f"1D time points should be ordered from low to high"
    assert t_list_1D[-1] == t_list_2D[
        -1], f"1D gradient program end time should be the same as the shifting gradient end time, got:" \
             f" {t_list_1D[-1], t_list_2D[-1]}"

    # get the 2D parameters from the settings dictionary
    t_M_2D = settings_2D['t_M']
    t_G_2D = settings_2D['t_G']
    t_init_2D = settings_2D['t_init']
    t_D_2D = settings_2D['t_D']
    t_0_2D = settings_2D['t_0']

    # check if there is a whole number of modulations that fits in the 1D program
    assert (t_list_2D[
                -1] / t_M_2D) % 1 == 0, f"There should be a integer number of modulations in the 1D gradient program, " \
                                        f"got: {t_list_2D[-1] / t_M_2D} modulations"
    # check if deadtime + dwelltime + inittime + gradient time is lower than the modulation time
    assert t_0_2D + t_D_2D + t_init_2D + t_G_2D < t_M_2D, f"deadtime + dwelltime + inittime + gradient time should be " \
                                                          f"lower than the modulation time, got:" \
                                                          f" {t_0_2D + t_D_2D + t_init_2D + t_G_2D} > {t_M_2D}"
    # check if the time between two shifting segments is larger than the modulation time
    for i in range(1, len(t_list_2D)):
        assert t_list_2D[i] - t_list_2D[i - 1] > t_M_2D, f"modulation time should be smaller than the time between two " \
                                                          f"shifting segments, got: {t_list_2D[i], t_list_2D[i - 1]} < " \
                                                          f"{t_M_2D}"
    return None


