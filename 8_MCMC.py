import argparse
import logging

import torch

import data
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

logging.basicConfig(format='%(message)s', level=logging.INFO)
pyro.enable_validation(__debug__)
pyro.set_rng_seed(0)


def construct_data():
    J = 8
    y = torch.tensor([28,  8, -3,  7, -1,  1, 18, 12]).type(torch.Tensor)
    sigma = torch.tensor([15, 10, 16, 11,  9, 11, 10, 18]).type(torch.Tensor)
    return {"J": J, "y": y, "sigma": sigma}

# our stocfun
# sigma is a constant you can pass in that describes the variance of our obs distribution
# eta, mu, tau are latent variables, we are interested in their posterior distributions
# after observing obs.
# Note: we don't use plate notation here because we assume we are observing many samples from
#       a single distribution P(obs|eta, mu, tau). This is analogous to if in our
#       titanic example, we made several outcome draws from a single person's data.
#       This is a subtle difference between having many different people and making
#       a single observation for each.

# Now I'm getting confused, what's the difference between a dataset full of images (VAE)
# where we need 1 node per instance, and this example where we use 1 node for all instances?

def model(J):

    def model_inner(sigma):

        eta = pyro.sample('eta', dist.Normal(torch.zeros(J), torch.ones(J)))
        mu = pyro.sample('mu', dist.Normal(torch.zeros(1), 10 * torch.ones(1)))
        tau = pyro.sample('tau', dist.HalfCauchy(scale=25 * torch.ones(1)))

        theta = mu + tau * eta

        return pyro.sample("obs", dist.Normal(theta, sigma))

    return model_inner


# poutine.condition allows us to apply observations onto a stocfun (so we don't need to)
# define the observations inside the stocfun itself
# DOCS
# Given a stochastic function with some sample statements and a dictionary of observations at names,
# change the sample statements at those names into observes with those values.
# params: stocfun, data, other params
# returns: a stocfun with data applied as observations, has the same arguments as original stocfun

def conditioned_model(model, sigma, y):
    return poutine.condition(model, data={"obs": y})(sigma)


def main(args):
    # define which MCMC algorithm to run (proposal, rejection, etc.)
    # this is captured by the notion of a "kernel"
    # NUTS: No-U-Turn Sampler kernel, which provides an efficient and convenient way
    # to run Hamiltonian Monte Carlo. The number of steps taken by the
    # integrator is dynamically adjusted on each call to ``sample`` to ensure
    # an optimal length for the Hamiltonian trajectory [1]. As such, the samples
    # generated will typically have lower autocorrelation than those generated
    # by the :class:`~pyro.infer.mcmc.HMC` kernel.

    nuts_kernel = NUTS(conditioned_model)

    # MCMC is the wrapper around the actual algorithm variant, you call  .run on it
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)

    data = construct_data()

    mcmc.run(model(data["J"]), data["sigma"], data["y"])
    mcmc.summary(prob=0.5)


if __name__ == '__main__':
    assert pyro.__version__.startswith('1.2.1')
    parser = argparse.ArgumentParser(description='Eight Schools MCMC')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='number of MCMC samples (default: 1000)')
    parser.add_argument('--num-chains', type=int, default=1,
                        help='number of parallel MCMC chains (default: 1)')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='number of MCMC samples for warmup (default: 1000)')
    args = parser.parse_args()

    main(args)