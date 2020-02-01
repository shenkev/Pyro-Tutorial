import math
from tqdm import tqdm
import torch
from torch.distributions import constraints

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist


# It seems like Pyro has been built with special focus on
# Stochastic Variational Inference. Presumably SVI is Uber ATG's inference algorithm of
# choice. There's also moderate support for MCMC, HMC, etc. but they've completely
# neglected rejection sampling and others. I guess these algorithms are not as practical
# and serve as more learning tools - which Uber doesn't care so much about.


# Some useful notes to understand the SVI API
# 1. names will be used to align the random variables in the model and guide (posterior)
# 2. guides are approximate posteriors
# 3. Pyro enforces that model() and guide() have the same call signature (take the same)
# arguments, which could be an empty set of course.
# 4. since guide is an approximation to the posterior pθmax(z|x), guide needs to provide
# a valid joint probability density over all the latent random variables in the model
# e.g.
# def guide():
#     pyro.sample("z_1", ...)
# def model():
#     pyro.sample("z_1", ...)


# The general variational inference setup as 2 ingredients.
# 1. the need to maximize the evidence P(x) factorized as Pθ(x|z)Pθ(z)
# 2. the need to model the posterior P(z|x) using a variational approximation qϕ(z|x)
# It turns out 1 and 2 conveniently can be achieved together by maximizing the ELBO
# Note: 1. is essentially learning the dynamics in the counterfactual context whereas
# 2. is learning an inverse dynamics. The former is useful for prediction, the second
# for abduction.
# Note: 1. is in general very hard because you want to find the dynamics parameters
# that maximize the dataset X, but you don't know which values of Z should map to which
# values of X. Hence you need to figure out 2 things at once: the parameters θ and the
# Z's which should map to the X's. This leads to trying to optimize an integral and is
# very intractable.

# one thing I'm confused about is where we pass in the dataset X, "guess" in tutorial 2
# was a dirac-delta prior rather than a data sample/observation
# another way to state this is, what if we observe a DISTRIBUTION for variable "measure"
# instead of a single value 1000/9.5?

# API Notes:
# 1. SVI is performed using the Pyro SVI class
# .step(args) takes one gradient step and returns the -ELBO
# args are passed to model/guide.
# .evaluate_loss() just returns the -ELBO, args are also passed tdo model/guide
# 2. pyro.optim is a wrapper around torch.optim and has the same optimizers
# adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
# optimizer = Adam(adam_params)


# A Simple Example
# Let's consider a toy experiment where we're trying to deduce the fairness of a coin.
# A brand new coin is likely fair (50% head and tails) whereas an old coin may be
# chipped or full of dirt on one side, and be slightly unfair (e.g. 55% heads).
# Variables
# Z - the fairness of the coin, defined as a real value on [0, 1]
# X - the outcome of a flip, defined as {0, 1}
# Prior Pθ(Z): Beta(a, b) which takes 2 integer "pseudo-count" parameters and returns
#              a real value sample. Actually a, b  can be real.
# Likelihood Pθ(x|z): Bernoulli(z) which takes a real value fairness parameter and
#                     returns a binary sample.
# VI Posterior qϕ(z|x): this is a distribution over fairness, which means it should
#                       be a distribution over real values, maybe a beta distrib
#                       makes sense

def model(alpha0=10.0, beta0=10.0):
    
    alpha = torch.tensor(alpha0)
    beta = torch.tensor(beta0)
    
    def model_inner(data):
    
        # sample f from the beta prior
        f = pyro.sample("z_fairness", dist.Beta(alpha, beta))

        # loop over the observed data        
        # this is pretty savage, we're creating one obs variable per data sample
        # z -> x1, z -> x2, z -> x3, ... isn't there a more efficient way?
        # each of the x1, x2, x3 have the same distribution

        for i in range(len(data)):
            pyro.sample("x_{}".format(i), dist.Bernoulli(f), obs=data[i])
    
    return model_inner

def guide(data):
    # Note: beta prior is the ‘right’ choice, b/c conjugacy of the bernoulli and beta
    # distributions means that the exact posterior is a beta distribution

    # why not guess the posterior is the prior?
    a = pyro.param("alpha_q", torch.tensor(15.0), constraint=constraints.positive)
    b = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive)

    pyro.sample("z_fairness", dist.Beta(a, b))  # to match the model definition

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# set up the optimizer
adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

# setup the inference algorithm
svi = SVI(model(), guide, optimizer, loss=Trace_ELBO())

# generate some fake data
NPos = 10
NNeg = 5

data = []
for _ in range(NPos):
    data.append(torch.tensor(1.0))
for _ in range(NNeg):
    data.append(torch.tensor(0.0))

n_steps = 2000
# do gradient steps
for step in tqdm(range(n_steps)):
    svi.step(data)

# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()

# here we use some facts about the beta distribution
# compute the inferred mean of the coin's fairness
inferred_mean = alpha_q / (alpha_q + beta_q)
# compute inferred standard deviation
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

print("True posterior based on counting real and pseudoflips: {}".format((NPos+10.0)/(NPos+NNeg+20.0)))