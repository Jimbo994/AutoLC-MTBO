import numpy as np
import matplotlib.pyplot as plt
import torch

from bo_code.BO import BO_round
from bo_code.MTBO import generate_initial_data_mtgp, MTBO_round

from rm_code.crf import resolution_score_2D
from rm_code.retention_model import online_system, offline_system, offline_system2
from rm_code.plot_chromatogram import plot_chromatogram, plot_contour_spectrum,  plot_shifting_2D_gradient

from utils.utils import bo_to_rm_2D, check_pars, seed_everything, best_so_far, ci

from tqdm import tqdm
import pandas as pd

tkwargs = (
    {  # Dictionary containing information about data type and data device
        "dtype": torch.double,
        "device": torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu"),
    }
)
# Writing the dictionary to a DataFrame
df = pd.DataFrame([tkwargs])
# Saving the DataFrame to a CSV file, this will be read by other functions.
df.to_csv('data/tkwargs.csv', index=False)

### Set up a 1D retention model

# Set some default parameters for first dimension
t_0_1D = 4.5 # dead time
t_D_1D = 0.1 # dwell time
N_1D = 400 # plate number 1D
t_init_1D = 2 # init time

# create dictionary for the above described parameters
settings_1D = {'t_0': t_0_1D, 't_D': t_D_1D, 'N': N_1D, 't_init': t_init_1D}

### Set up a 2D retention model

### Define 2D parameters
# lower bounds of shifting gradients are named "init". upper bounds of shifting gradients aer named "final".
# time points of each shift have to be same in the current setup. Shift starts after the dead time of the first dimension. Shift ends at the end of the 1D program
# each gradient in the second dimension has a dead time, gradient time and modulation time.

N_2D = 100000 # plate number 2D

t_M_2D = 2 #/ 3 # modulation time minutes
t_G_2D = 1.8 #/ 3 # gradient time minutes
t_init_2D = 0.1 #/ 3 # init time minutes
t_D_2D = 0.01 # dwell time in minutes
t_0_2D = 0.01 # dead time in minutes

# create dictionary for the above described parameters
settings_2D = {'t_M': t_M_2D, 't_G': t_G_2D, 't_init': t_init_2D, 't_D': t_D_2D, 't_0': t_0_2D, 'N': N_2D}

# maximum allowed times in first and second dimension
max_T = [100, t_M_2D]

# number of analytes
n_analytes = 80

# Load the retention parameters created in "samplin_retention_paremters.ipynb"
ret_pars = pd.read_csv('data/retention_system.csv').to_dict(orient='list')

# Optimization code

# Set some fixed parameters
t_max = 100 # maximum time
phi_min, phi_max = 0, 1 # maximum phi
fixed_phi_pars_1D = torch.tensor([[phi_min], [phi_max]]) # fixed phi points
fixed_time_pars_1D = torch.tensor([[0.], [t_max]]) # at fixed time points

fixed_phi_pars_2D = torch.tensor([[phi_min], [phi_max]]) # fixed phi points
fixed_time_pars_2D = torch.tensor([[0.],[t_0_1D], [t_max]]) # at fixed time points

# We will optimize 12 parameters, 4 parameters (2 gradient turning points) in the first dimension.
# And 8 parameters in the second dimension, two time points, and three phi_init and phi_final point of the shifting gradient.
# Pars vector will look like this: [phi1, phi2, t1, t2, phi_i1, phi_i2, phi_i3, phi_f1, phi_f2, phi_f3, t1_shift, t2_shift]
bounds = torch.stack([
    torch.tensor([phi_min, phi_min, 0.1, 0.1, phi_min, phi_min, phi_min, phi_min, phi_min, phi_min, t_0_1D, t_0_1D]),
    torch.tensor([phi_max, phi_max, t_max-0.1, t_max-0.1, phi_max, phi_max, phi_max, phi_max, phi_max, phi_max, t_max-t_M_2D, t_max-t_M_2D])]
)

# bounds after normalization to [0,1]
norm_bounds = torch.stack([torch.zeros(12), torch.ones(12)])

# We will also need to set some inequality constraints : 1. phi_i1 < phi_f2, 2. phi_i2 < phi_f3,
# 3. t1 < t2, 4. phi1 < phi2,  5. -t1_shift + t2_shift > tM
# Spare representation of BoTorch.
# ([indices of parameters], [coefficients], constant), example: ((torch.tensor([0,1]), torch.tensor([-1., 1.]), 0.0)) -x0 + x1 >= 0
inequality_constraints= [(torch.tensor([0,1]), torch.tensor([-1., 1.]), torch.tensor(0.0)), (torch.tensor([2,3]), torch.tensor([-1., 1.]), torch.tensor(0.1)), (torch.tensor([10,11]), torch.tensor([-1., 1.]), torch.tensor(t_M_2D)), (torch.tensor([4,8]), torch.tensor([-1., 1.]), torch.tensor(0.0)), (torch.tensor([5,9]), torch.tensor([-1., 1.]), torch.tensor(0.0))]

# draw 10 random indices between 0 and n_analytes
remove_indices = np.random.randint(0, n_analytes, 15)
# create dictionary with noise levels
noise = {'tR_1D': 2, 'tR_2D': 0.3, 'W_1D': 0.2, 'W_2D': 0.05}

# Set up an online only BO loop

# generate initial samples
n_init_online = 10
n_init_offline = 20

# optimization budget
iterations = 75
n_online = 1
n_offline=20

#variations = [5,10,20,40,50]
# number of trials
trials = 10

#for n_offline in variations:
# create some lists to store results in
scores_all_single_task = []
scores_all_online_mt = []
scores_all_offline_mt = []
scores_all_random = []

pars_all_single_task = []
pars_all_online_mt = []
pars_all_offline_mt = []
pars_all_random = []

# Loop over trials
for trial in range(trials):
    # print trial number out of total
    print('Trial', trial+1, 'out of', trials)

    # Set random seed for reproducibility
    seed_everything(trial)

    # create lists to fill with results per trial
    scores_single_task = []
    scores_online_mt = []
    scores_offline_mt = []
    scores_random = []

    pars_single_task = []
    pars_online_mt = []
    pars_offline_mt = []
    pars_random = []

    # generate parameters for initial experiments
    pars_online_init, pars_offline_init = generate_initial_data_mtgp(n_init_online, n_init_offline, bounds, inequality_constraints)

    # Perform initial online experiments
    phi_list_1D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D = bo_to_rm_2D(pars_online_init, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)

    for i in range(len(pars_online_init)):

        tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = online_system(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_init_2D[i], phi_final_2D[i], t_list_2D[i], max_T)

        # now we need to add the pars and scores to online_mt and single_task as they will both share the same initial experiments
        scores_online_mt.append(res_score - 0.1*time_score)
        pars_online_mt.append(pars_online_init[i])

        scores_single_task.append(res_score - 0.1*time_score)
        pars_single_task.append(pars_online_init[i])

        scores_random.append(res_score - 0.1*time_score)
        pars_random.append(pars_online_init[i])

    # Perform initial offline experiments
    phi_list_1D, t_list_1D, phi_init_2D, phi_final_2D, t_list_2D = bo_to_rm_2D(pars_offline_init, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)

    for i in range(len(pars_offline_init)):

        tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = offline_system2(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_init_2D[i], phi_final_2D[i], t_list_2D[i], max_T, noise, remove_indices)

        # now we need to add the pars and scores to offline_mt
        scores_offline_mt.append(res_score - 0.1*time_score)
        pars_offline_mt.append(pars_offline_init[i])

        #print(res_score)
    print('Done with initial random experiments')

    print('Starting Random Strategy')
    #RANDOM STRATEGY
    for iteration in tqdm(range(iterations)):
        # draw random parameters
        new_pars_random, _ = generate_initial_data_mtgp(n_online, n_offline, bounds, inequality_constraints)
        # convert to parameters that retention modeling code can handle
        phi_list_1D, t_list_1D, phi_list_init, phi_list_final, t_list_2D = bo_to_rm_2D(new_pars_random, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)

        for i in range(len(new_pars_random)):
            # Perform new experiment
            tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = online_system(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_list_init[i], phi_list_final[i], t_list_2D[i], max_T)

            # now we need to add the pars and scores to online_mt
            scores_random.append(res_score - 0.1*time_score)
            pars_random.append(new_pars_random[i])

    print('Starting Single Task BO loop')
    # SINGLE TASK BO
    for iteration in tqdm(range(iterations)):
        # perform BO round
        new_pars_so = BO_round(bounds, norm_bounds, inequality_constraints, scores_single_task, pars_single_task, n_online)
        #print(len(new_pars_so))
        # convert to parameters that retention modeling code can handle
        phi_list_1D, t_list_1D, phi_list_init, phi_list_final, t_list_2D = bo_to_rm_2D(new_pars_so, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)

        for i in range(len(new_pars_so)):
            # Perform new experiment
            tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = online_system(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_list_init[i], phi_list_final[i], t_list_2D[i], max_T)


            # now we need to add the pars and scores to online_mt
            scores_single_task.append(res_score - 0.1*time_score)
            pars_single_task.append(new_pars_so[i])

    # MULTI TASK BO
    #Now this will look different, as it will be a combination of online and offline experiments, and they need to be evaluated in the function.
    # print('Starting Multi Task BO loop')
    # for iteration in tqdm(range(iterations)):
    #     # perform MTBO round and query for n_offline experiments. These are obtained by optimizing an acquisition function on the online model.
    #     new_pars_mt_offline = MTBO_round(scores_online_mt, scores_offline_mt, pars_online_mt, pars_offline_mt, n_online, n_offline, bounds, inequality_constraints, mode="offline")
    #
    #     #print('NEW PARS ', new_pars_mt_offline.shape)
    #     # convert to parameters that retention modeling code can handle
    #     phi_list_1D, t_list_1D, phi_list_init, phi_list_final, t_list_2D = bo_to_rm_2D(new_pars_mt_offline, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)
    #
    #     # perform experiments on the offline system
    #     for i in range(len(new_pars_mt_offline)):
    #         # Perform new experiment
    #         tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = offline_system(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_list_init[i], phi_list_final[i], t_list_2D[i], max_T, noise, remove_indices)
    #
    #         # now we need to add the pars and scores to offline_mt
    #         scores_offline_mt.append(res_score - 0.1*time_score)
    #         pars_offline_mt.append(new_pars_mt_offline[i])
    #
    #
    #     # Now we need to update the MTGP model and check which design point is best on the online model posterior using the UCB acquisition function
    #     new_pars_mt_online = MTBO_round(scores_online_mt, scores_offline_mt, pars_online_mt, pars_offline_mt, n_online, n_offline,  bounds, inequality_constraints, mode="online")
    #
    #     #print('NEW PARS ', new_pars_mt_online.shape)
    #     #print('NEW PARS ', new_pars_mt_online)
    #     # convert to parameters that retention modeling code can handle
    #     phi_list_1D, t_list_1D, phi_list_init, phi_list_final, t_list_2D = bo_to_rm_2D(new_pars_mt_online, fixed_phi_pars_1D, fixed_time_pars_1D, fixed_phi_pars_2D, fixed_time_pars_2D)
    #
    #     # perform this experiment on the online system
    #     for i in range(len(new_pars_mt_online)):
    #         # Perform new experiment
    #         tR_list_1D, W_list_1D, tR_list_2D, W_list_2D, res_score, time_score = online_system(ret_pars, settings_1D, settings_2D, phi_list_1D[i], t_list_1D[i],phi_list_init[i], phi_list_final[i], t_list_2D[i], max_T)
    #
    #         # now we need to add the pars and scores to online_mt
    #         scores_online_mt.append(res_score - 0.1*time_score)
    #         pars_online_mt.append(new_pars_mt_online[i])

    # update lists
    scores_all_single_task.append(scores_single_task)
    # scores_all_online_mt.append(scores_online_mt)
    # scores_all_offline_mt.append(scores_offline_mt)
    scores_all_random.append(scores_random)
    #
    pars_all_single_task.append(pars_single_task)
    # pars_all_online_mt.append(pars_online_mt)
    # pars_all_offline_mt.append(pars_offline_mt)
    pars_all_random.append(pars_random)

# save results
np.save('data/scores_all_single_task.npy', scores_all_single_task)
np.save('data/scores_all_random.npy', scores_all_random)

np.save('data/pars_all_single_task.npy', pars_all_single_task)
np.save('data/pars_all_random.npy', pars_all_random)
