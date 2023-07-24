import torch
from botorch import fit_gpytorch_model
from botorch.acquisition import LinearMCObjective, qExpectedImprovement

from botorch.models import MultiTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf, gen_batch_initial_conditions
from botorch.sampling import IIDNormalSampler

from botorch.utils.sampling import get_polytope_samples
from gpytorch import ExactMarginalLogLikelihood

import pandas as pd

# Reading the CSV file back into a DataFrame
df_read = pd.read_csv('data/tkwargs.csv')

# Converting DataFrame back to a dictionary
tkwargs = {
    'dtype':eval(df_read['dtype'][0]),
    'device': torch.device(df_read['device'][0])
}

def generate_initial_data_mtgp(n_online, n_offline, bounds, inequality_constraints):
    """ Generates the initial data for the online and offline tasks, uses polytope sampling to generate the data

    Args:
        n_online (int): Number of data points for the online task
        n_offline (int): Number of data points for the offline task
        bounds (list): List of tuples containing the bounds for each dimension of the data
        inequality_constraints (list): List of tuples containing the inequality constraints for each dimension of the data

    Returns:
        train_X_online (Tensor): The data points for the online task
        train_X_offline (Tensor): The data points for the offline task
    """

    # check if n_offline is bigger than n_online and return warning if it is
    if n_offline < n_online:
        print("Warning: n_offline is smaller than n_online. n_offline will be set to n_online")
        n_offline = n_online

    for i in range(len(inequality_constraints)):
        tensor1, tensor2, *rest = inequality_constraints[i]
        inequality_constraints[i] = (tensor1.to(tkwargs['device']), tensor2.to(tkwargs['device']), *rest)

    bounds = bounds.to(tkwargs['device'])

    train_X_offline = get_polytope_samples(n_offline, bounds, inequality_constraints)

    # generate subset of train_X_offline
    train_X_online = train_X_offline[:n_online]

    # old code that adds a task feature to the data, but this is not needed for the current implementation
    #i1, i2 = torch.zeros(n_online, 1, **tkwargs), torch.ones(n_offline, 1, **tkwargs)
    #train_X_full = torch.cat([torch.cat([train_X_online, i1], -1), torch.cat([train_X_offline, i2], -1)])
    #train_Y_full = torch.cat((train_Y_online, train_Y_offline)).unsqueeze(-1)
    return train_X_online.detach().cpu(), train_X_offline.detach().cpu()

def remove_task_feature(train_X_full):
    """
    Removes the task feature from the data
    Args:
        train_X_full (Tensor): The data points with the task feature
    Returns:
        train_X (Tensor): The data points without the task feature
    """
    train_X = train_X_full[:, :-1]
    return train_X

def add_task_feature(train_X, task_feature):
    """
    Adds a task feature to the data. The task feature is an integer that is repeated for each data point.
    Args:
        train_X (Tensor): The data points
        task_feature (int): The task feature to add to the data points
    Returns:
        train_X_full (Tensor): The data points with the task feature added
    """
    task_feature = torch.tensor(task_feature, **tkwargs).repeat(train_X.shape[0], 1)
    train_X_full = torch.cat([train_X, task_feature], -1)
    return train_X_full

def construct_mtgp_acqf(model, train_X_full, train_Y_full, num_samples=128):
    """Constructs the acquisition function for the MTGP model.
    Args:
        model: The MTGP model.
        train_X_full (Tensor): The training features. Dimensions are `n x d + 1`, last dimension is task feature.
        train_Y_full (Tensor): The training observations. Dimensions are `(2*n) x m`.
        num_samples (int): The number of samples to use for the MC sampler.
    Returns:
        The acquisition function.
        """
    # create objective with full focus on online task
    objective = LinearMCObjective(weights=torch.tensor([1.0, 0], **tkwargs))
    # create the acq function
    sampler = IIDNormalSampler(sample_shape=torch.Size([num_samples])).to(**tkwargs)
    acq_func = qExpectedImprovement(
        model=model, best_f=train_Y_full[train_X_full[:, -1] == 0].max().item(), sampler=sampler, objective=objective,
    )#.to(**tkwargs)
    return acq_func

def optimize_acqf_and_get_observation_offline(acq_func, bounds, inequality_constraints, q, num_restarts=20, raw_samples=512):
    """Optimize the acquisition function, and return new candidates.
    Args:
        acq_func: The acquisition function.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        inequality_constraints: A tuple of two lists of callables. The first list contains inequality constraints of
            the form `g_i(X) >= 0`, while the second list contains equality constraints of the form `h_i(X) = 0`.
        q: The number of candidates to generate.
        num_restarts: The number of restarts for the local optimization procedure.
        raw_samples: The number of samples used to initialize the local optimization procedure.
    Returns:
        new_x: A `q x d+1`-dim tensor of `q` new candidates. last column is the task feature.
        """
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        inequality_constraints=inequality_constraints,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        sequential=True,
        #timeout_sec = 360,
    )
    # observe new values
    new_x = candidates.detach()
    # add offline task feature using add task feature
    #new_x = add_task_feature(new_x, task_feature=1)
    return new_x

def MTBO_round(scores_online, scores_offline, pars_online, pars_offline, n_online, n_offline, bounds, inequality_constraints, mode="offline"):
    """

    """
    train_X_online = torch.stack(pars_online).to(**tkwargs)
    train_X_offline = torch.stack(pars_offline).to(**tkwargs)

    train_Y_online = torch.tensor(scores_online).unsqueeze(-1).to(**tkwargs)
    train_Y_offline = torch.tensor(scores_offline).unsqueeze(-1).to(**tkwargs)

    # Create the full rank data from the online and offline data
    train_X_online = add_task_feature(train_X_online, 0)
    train_X_offline = add_task_feature(train_X_offline, 1)

    train_X_full = torch.cat([train_X_online, train_X_offline])
    train_Y_full = torch.cat([train_Y_online, train_Y_offline])

    for i in range(len(inequality_constraints)):
        tensor1, tensor2, *rest = inequality_constraints[i]
        inequality_constraints[i] = (tensor1.to(tkwargs['device']), tensor2.to(tkwargs['device']), *rest)

    bounds = bounds.to(tkwargs['device'])

    # initialize model
    model = MultiTaskGP(train_X_full, train_Y_full, task_feature=-1, outcome_transform=Standardize(m=1),
                        input_transform=Normalize(d=train_X_full.shape[-1], bounds=bounds,
                        indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))#.to(**tkwargs)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # fit model
    fit_gpytorch_model(mll)

    if mode == "offline":
        #print('offline')
        # construct acquisition function
        acq_func = construct_mtgp_acqf(model, train_X_full, train_Y_full)#.to(**tkwargs)

        # print('generating initial conditions')
        # X_init = gen_batch_initial_conditions(acq_func, bounds, q=n_offline, num_restarts=20, raw_samples=512,
        #                                       inequality_constraints=inequality_constraints)
        # print('done')

        # print('optimizing offline acq func' )
        # optimize acquisition function and get new observation
        new_x = optimize_acqf_and_get_observation_offline(acq_func, bounds=bounds,
                    inequality_constraints=inequality_constraints, q=n_offline, num_restarts=10, raw_samples=256)
        # print('done')
        return new_x.detach().cpu()

    if mode == "online":
        #print('online')
        # take n_offline last points from train_X_full and remove the task feature from them
        # Then compute the posterior of the online model and use UCB to pick the index with the highest value
        new_y = model.posterior(remove_task_feature(train_X_full[-n_offline:]), [0]).mean #+ \
        #        0.3 * model.posterior(remove_task_feature(train_X_full[-n_offline:]), [0]).variance.sqrt()

        # obtain index and values of N_ONLINE highest values
        values, indices = torch.topk(new_y, n_online, dim=0)

        new_x = train_X_full[-n_offline:][indices].squeeze(1)
        new_x = remove_task_feature(new_x)
        return new_x.detach().cpu()
