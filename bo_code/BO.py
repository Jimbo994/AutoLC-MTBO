import torch
from botorch.utils.sampling import get_polytope_samples

try:
    from botorch.sampling.samplers import SobolQMCNormalSampler
except:
    from botorch.sampling.normal import SobolQMCNormalSampler, IIDNormalSampler

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qNoisyExpectedImprovement

from botorch.optim.optimize import optimize_acqf

import pandas as pd

# Reading the CSV file back into a DataFrame
df_read = pd.read_csv('data/tkwargs.csv')

# Converting DataFrame back to a dictionary
tkwargs = {
    'dtype':eval(df_read['dtype'][0]),
    'device': torch.device(df_read['device'][0])
}

def generate_initial_data(n, bounds, inequality_constraints):
    return get_polytope_samples(n, bounds, inequality_constraints)

def BO_round(bounds, norm_bounds, inequality_constraints, scores, pars, n_online):
    """
    This function fits a Gaussian process to a single objective.
    Then uses the qNoisyExpectedImprovement acquisition function to get experiment to perform next.

    Args:
        bounds: bounds of optimizable parameters
        norm_bounds: normalized bounds of optimizable (i.e. [0,1]^N)
        scores: list of previously obtained scores, i.e. train_Y
        pars: list of previously performed parameters, i.e, train_X
        n_online: int, number of candidates to return.

    Returns:
        candidate, the parameters to evaluate next.
    """

    train_X = torch.stack(pars).to(**tkwargs)
    train_Y = torch.tensor(scores, **tkwargs).unsqueeze(-1)

    for i in range(len(inequality_constraints)):
        tensor1, tensor2, *rest = inequality_constraints[i]
        inequality_constraints[i] = (tensor1.to(tkwargs['device']), tensor2.to(tkwargs['device']), *rest)

    bounds = bounds.to(tkwargs['device'])

    # Initialize Gaussian Process and MarginalLogLikelihood
    gp = SingleTaskGP(train_X, train_Y, input_transform=Normalize(d=len(bounds[0]), bounds=bounds), outcome_transform=Standardize(m=1)).to(**tkwargs)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

    # Fit parameters
    fit_gpytorch_model(mll)

    sampler = IIDNormalSampler(sample_shape=torch.Size([128])).to(**tkwargs)

    qNEI = qNoisyExpectedImprovement(gp, train_X, sampler).to(**tkwargs)

    # Optimize acq. fun.
    candidate, acq_value = optimize_acqf(
        qNEI, bounds=bounds, inequality_constraints=inequality_constraints, q=n_online, num_restarts=10, raw_samples=256, sequential=True
    )

    return candidate.detach().cpu()
