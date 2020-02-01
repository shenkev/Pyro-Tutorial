import pyro
import pyro.distributions as dist
from pyro.optim import Adam
from pyro.infer import SVI, TraceGraph_ELBO

from torch.distributions import constraints
import torch.nn as nn
import torch
from tqdm import tqdm
import math

# This section starts with some theoretical explaination of gradient estimators.
# In the general case we want to optimize an expression like E[f(x, z)].
# In reinforcement learning, f = R is the reward function. In variational
# inference, f = ELBO is the lower bound we're trying to maximize.

# The basic conclusion is that if f() is reparameterizable (e.g. Gaussian), then
# we can use the reparameterization trick, else we use REINFORCE which has high
# variance. Reparam trick is usually possible for exponential family continuous
# distributions while discrete distributions usually need REINFORCE (I thought)
# there was a paper that derived the reparam version?

# The discussion then goes into tricks for variance reduction, including
# exploiting the structure of the graphical model. The main practitioner's
# conclusion is all of this is taken care of inside the TraceGraph_ELBO() loss.
# leveraging this dependency information takes extra computations, so
# TraceGraph_ELBO should only be used when your model has non-reparameterizable
# random variables; in most applications Trace_ELBO suffices.


# Plate notation is not useless
# so for some reason Pyro cannot figure out graph independency itself, so if you
# don't mark them using "plate", then it can't take advantage of these independ.
# to reduce the variance of the estimators.

# ks = pyro.sample("k", dist.Categorical(probs).to_event(1))
# pyro.sample("obs", dist.Normal(locs[ks], scale).to_event(1),obs=data)
# this is bad and will use a naive estimator

# (assumed to be along the rightmost tensor dimension)
# with pyro.plate("foo", data.size(-1)):
#     ks = pyro.sample("k", dist.Categorical(probs))
#     pyro.sample("obs", dist.Normal(locs[ks], scale), obs=data)
# this is good and will use fancy Rao-Blackwellization estimator

# A last remark on plating: the fact we need to use it is a consequence of how
# Pyro currently does dependency inferring.
# we expect to add finer/better dependency tracking in a future version of Pyro
# sometimes it pays to reorder random variables within a stocfun (if possible)


# Baselines when using Reparam Trick
# z = pyro.sample("z", dist.Bernoulli(...),
#  infer=dict(baseline={'use_decaying_avg_baseline': True, 'baseline_beta': 0.95}))
# it's very easy to specify a baseline


# Neural Baselines
# neural network baselines have built-in support

class BaselineNN(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(BaselineNN, self).__init__()
        self.linear = nn.Linear(dim_input, dim_hidden)
        # ... finish initialization ...

    def forward(self, x):
        hidden = self.linear(x)
        # ... do more computations ...
        return baseline
# first define a neural network

def guide(x):  # here x is the current mini-batch of data
    pyro.module("my_baseline", baseline_module)
    # ... other computations ...
    z = pyro.sample("z", dist.Bernoulli(...),
                    infer=dict(baseline={'nn_baseline': baseline_module,
                                         'nn_baseline_input': x}))

# simply specify a nn baseline in the stocfun
# note the model needs to be "registered" with Pyro using pyro.module for
# autograd to work. The default is for the baseline model to take 1 grad
# step everytime the inference parameters take 1 autograd step

def per_param_args(module_name, param_name):
    if 'baseline' in param_name or 'baseline' in module_name:
        return {"lr": 0.010}
    else:
        return {"lr": 0.001}

# optimizer = optim.Adam(per_param_args)
# controlling the learning rate


# b = # do baseline computation
# z = pyro.sample("z", dist.Bernoulli(...),
#                 infer=dict(baseline={'baseline_value': b}))
# you can choose to avoid the Pyro API for doing baseline gradient
# updates and instead implement your custom computation of baseline values
# and updatin the networks


# Reimplementation of 3_SVI_tensorized.py with a REINFORCE objective and baseline
# Note: our posterior beta distribution is actually reparamable but we can use
# REINFORCE just for the purpose of illustrating the effectiveness of baselines

from pyro.distributions.testing.fakes import NonreparameterizedBeta
# it turns out Pyro handles figuring out if a distribution is reparamable or not
# in this case, we need to import an artificial implementation of beta that isn't


def model(alpha0=10.0, beta0=10.0):
    
    alpha = torch.tensor(alpha0)
    beta = torch.tensor(beta0)
    
    def model_inner(data):
    
        f = pyro.sample("z_fairness", dist.Beta(alpha, beta))
        # f_tensor = f*torch.ones(data.shape)  <--- not necessary

        with pyro.plate("data_plate"):
            pyro.sample("x", dist.Bernoulli(f), obs=data)

    
    return model_inner


def do_inference(use_baseline=True):

    def guide(data):

        a = pyro.param("alpha_q", torch.tensor(15.0), constraint=constraints.positive)
        b = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive)

        baseline = dict(baseline={'use_decaying_avg_baseline': use_baseline, 'baseline_beta': 0.9})
        pyro.sample("z_fairness", NonreparameterizedBeta(a, b), infer=baseline)


    # Inference
    pyro.clear_param_store()

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    alpha0, beta0 = 10.0, 10.0
    svi = SVI(model(alpha0, beta0), guide, optimizer, loss=TraceGraph_ELBO())

    NPos = 10
    NNeg = 5
    data = torch.tensor(NPos*[1.0] + NNeg*[0.0])

    def param_abs_error(name, target):
        return torch.sum(torch.abs(target - pyro.param(name))).item()

    # True parameters

    true_alpha = data.sum() + alpha0
    true_beta = len(data)-data.sum() + beta0

    # Run

    n_steps = 10000
    for step in tqdm(range(n_steps)):
        svi.step(data)

        # compute the distance to the parameters of the true posterior
        alpha_error = param_abs_error("alpha_q", true_alpha)
        beta_error = param_abs_error("beta_q", true_beta)

        # stop inference early if we're close to the true posterior
        if alpha_error < 0.8 and beta_error < 0.8:
            break

    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    inferred_mean = alpha_q / (alpha_q + beta_q)
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)

    print("Parameters after {} steps: {} {} {} {}".format(step, true_alpha, true_beta, alpha_q, beta_q))
    print("\nbased on the data and our prior belief, the fairness " +
        "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
    print("True posterior based on counting real and pseudoflips: {}".format((NPos+10.0)/(NPos+NNeg+20.0)))


do_inference(use_baseline=True)
do_inference(use_baseline=False)  # converges much slower