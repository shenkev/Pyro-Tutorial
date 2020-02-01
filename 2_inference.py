import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from torch.distributions import constraints
from tqdm import tqdm

import pyro
from pyro.infer import Importance
import pyro.optim as optim
import pyro.distributions as dist


# This is the interesting part, we've defined the atoms of our system (stocfuns) or
# data-generating functions (and their higher-order variants), we can now rely on Pyro
# to do inference over the implicit probability distributions those stocfuns represent.

# we start with a simple physics problem setup:
# we can guess the weight of an object, there's a noisy measurement 
# guess --> weight --> measurement
# this graph is pretty nonsensical, guess <-- weight --> measurement would make more sense

def get_value(x):
    if type(x) == float:
        return x
    else:
        return x.item()


def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
    measure = pyro.sample("measure", dist.Normal(weight, 0.75))
    return guess, weight, measure


# The abstract "Recipe" of inference is to condition a generative model on observed data or
# evidence and infer the latent factors that might have produced those observations.
# This is implmented in Pyro by feeding in observed variables/evidence to a stocfun.
# The input and most importantly output shape of the stocfun remains the same, the difference
# is the observed variables are fixed to their observed values and latent variables are
# being sampled from their posteriors (after seeing observations) rather than priors.
# Interestingly, we are again remaining in the realm of samples as implicit representations
# of distributions, even for the posterior distributions after inference.
# I guess one question is, how do we visualize our posterior distribution? (to come)

# e.g. what is P(weight | guess = 8.5,  measure = 9.5)
# the hidden assumption is that your guess is actually quite good, weight is always distributed
# uniformly around it with variance 1 (your guess can't be way off), that's why your guess
# is valid evidence in this model
# The function pyro.condition allows us to feed evidence to the generative model

conditioned_scale = pyro.condition(scale, data={'measure': 9.5})

def condition_wrapper(measurement=9.5):
    return pyro.condition(scale, data={'measure': measurement})

# first arg is the stocfun, second is the observations
# the return type is another stocfun with the same input as scale (guess) and same output
# notice how the sample() names are essentially node names for the DAG


# when you're defining the original stocfun, the input arguments are the causal exogenous
# variables because that's where it makes sense to start the causal dynamics definitions
# however, evidence can of course be any variable in the graph, this is reflected is us
# wrapping the posterior distribution with a function that takes measurement evidence as
# input
def deferred_conditioned_scale(measurement, guess):
    return pyro.condition(scale, data={"measure": measurement})(guess)


# you may find this kind of laborous, alternatively Pyro allows you to define posteriors
# distributions directly by fixing the values of variables in the sample() definition
# this is done using the "obs" key word

def scale_obs(guess, measurement=9.5):  # equivalent to conditioned_scale above
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
     # here we condition on measurement == 9.5
    measure = pyro.sample("measure", dist.Normal(weight, 0.75), obs=measurement)
    return guess, weight, measure

# The exciting thing is Pyro comes with a "do" function for causal inference, it works exactly
# the same semantically as "condition" but under the hood it implements intervention rather
# than conditioning.
# Note however, this will only save the "intervention step" of counterfactual computation, the
# "do" function cannot compute other-world scenarios out of the box like 
#      ---- Y in the world where we do(T=1) given Y=5 in the world where we do(T=0) ----
# we still have to do abduction and prediction ourselves.

# "do" is unsupported for direct definitions of posteriors/augmented graphs
# def scale_intervene(guess, measurement=9.5):
#     weight = pyro.sample("weight", dist.Normal(guess, 1.))
#     return pyro.sample ("measure", dist.Normal(weight, 0.75), do=ERROR)

# here we intervene on the measuremnet == 9.5 which should not affect P(weight)
intervened_scale = pyro.do(scale, data={'measure': 9.5})

def intervene_wrapper(measurement=9.5):
    return pyro.do(scale, data={'measure': measurement})

# Unfortunately this isn't the end of the story
# Pyro isn't SO plug-and-play that our posterior stocfuns are now ready to draw samples from
# the posterior distribution. Unfortunateley, we still need to specify the nitty-gritty
# of the inference algorithm and "press run".

# Inference algorithms:
# importance sampling, rejection sampling, sequential Monte Carlo, MCMC
# and independent Metropolis-Hastings, and as variational distributions
# These all require as input some notion of a "proposal distribution" e.g. for the random walk
# in MCMC or the proposal for rejection sampling. Or the variational distribution for VI.
# We need to (intelligently) choose the proposal distribution.
# Note: the exact definition of what makes a good proposal distribution varies between each
# algorithm, but Pyro has tried to unified this notion of proposal distrib into a "guide"
# Usually you want to choose the "guide" or proposal distribution that is as similar as possible
# to the true posterior, then there is "less work for the model to do".

# in our example above, there is a closed form solution for the posterior, let's make that the
# guide

def true_mean_std(guess, measurement):
    return (0.75**2 * guess + measurement) / (1 + 0.75**2), np.sqrt(0.75**2/(1 + 0.75**2))

def perfect_guide(measurement=9.5):
    mean, std = true_mean_std(guess, measurement)
    return lambda guess: pyro.sample("weight",  dist.Normal(mean, std))


def perfect_intervention_guide(guess):
    return pyro.sample("weight", dist.Normal(guess, 1))


# Let's do inference using importance sampling https://www.wikiwand.com/en/Importance_sampling

def run_IS(model, guide, guess, samples=100, posterior_var='weight'):

    is_posterior = Importance(model, guide=guide, num_samples=samples).run(guess)
    is_marginal = pyro.infer.EmpiricalMarginal(is_posterior, posterior_var)
    is_samples = np.asarray([is_marginal().detach().item() for _ in range(samples)])
    print("mean: {0:.2f}, std: {1:.2f}".format(is_samples.mean(), is_samples.std()))
    return is_samples


# Let's plot some of these posteriors
# recall the DAG: guess --> weight --> measuremnet

def plot3D(data):
    fig = plt.figure()
    ax = Axes3D(fig)
    guess, weight, measure = data[:, 0], data[:, 1], data[:, 2]
    ax.scatter(guess, measure, weight)
    ax.set_xlabel('guess')
    ax.set_ylabel('measure')
    ax.set_zlabel('weight')
    plt.show()

doIS = False
if doIS:

    N = 100

    # guess5_post = np.asarray([scale(5.) for i in range(N)])
    # print("mean: {0:.2f}, std: {1:.2f}".format(guess5_post[:,1].mean(), guess5_post[:,1].std()))
    # plot3D(guess5_post)

    # marginal = guess5_post.copy()
    # marginal[:, 2] = 0
    # plot3D(marginal)

    guess_obs = 5.
    measure_obs = 1000
    guess5_measure95_post = run_IS(condition_wrapper(measure_obs), perfect_guide(measure_obs),
     guess=guess_obs, samples=N)
    plot3D(np.stack([guess_obs*np.ones(N), guess5_measure95_post, measure_obs*np.ones(N)], axis=1))

    guess_obs = 5.
    measure_int = 1000
    # this is the correct answer
    guess5_domeasure95_post = run_IS(intervene_wrapper(measure_int), perfect_intervention_guide,
     guess=guess_obs, samples=N)
    # this is the incorrect answer and shows the problem with importance sampling with bad guides
    wrong = run_IS(intervene_wrapper(measure_int), perfect_guide(9.5), guess=guess_obs, samples=N)
    plot3D(np.stack([guess_obs*np.ones(N), guess5_domeasure95_post, measure_int*np.ones(N)], axis=1))


# Variational Inference
# Everything has worked decently thus far except the fact that importance sampling broke down when
# the guide function was not good enough.
# VI is essentially the technique whereby instead of specifying a guide, you specify a family of
# functions to approximate the posterior instead, your "guide" is the loss function that pushes
# the variational model towards the true posterior. The parameters of the model are learned.
# Note: VI is not the same as fitting the dynamics in computing countefactuals, it's a choice of
# inference technique, we can potentially use VI in abduction.

# The paradigmn of VI requires to define an approximate posterior distribution (in this case
# Gaussian). The approximate posterior is now considered the "guide". The guide is no longer
# a proposal distribution but an approximation to the true posterior with learnable params.

def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(1.))  # guess is a required param but you don't need to use it
    # a = pyro.param("a", torch.tensor(guess))  # not necessary to use the guess here
    b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, b))
# we are explicitly defining a distribution for variable "weight", notice how Pyro works by
# matching the variable name "weight" in the approx-post to the "weight" in the data generating
# stocfun.

def run_SVI(model, guide, guess=5, num_steps=2500, lr=0.001, mom=0.1):
    # make sure all learnable parameters are reset
    pyro.clear_param_store()
    # 
    svi = pyro.infer.SVI(model=model,
                        guide=guide,
                        optim=optim.SGD({"lr": lr, "momentum": mom}),
                        loss=pyro.infer.Trace_ELBO())


    losses, a,b  = [], [], []
    
    for _ in tqdm(range(num_steps)):
        losses.append(svi.step(guess))
        a.append(pyro.param("a").item())
        b.append(pyro.param("b").item())

    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss")
    print('a = ', pyro.param("a").item())
    print('b = ', pyro.param("b").item())
    plt.show()

    plt.subplot(1,2,1)
    plt.plot([0, num_steps],[9.14, 9.14], 'k:')
    plt.plot(a)
    plt.ylabel('a')

    plt.subplot(1,2,2)
    plt.ylabel('b')
    plt.plot([0,num_steps],[0.6, 0.6], 'k:')
    plt.plot(b)
    plt.tight_layout()

    plt.show()

doSVI = True
if doSVI:

    guess_obs = 5.
    measure_obs = 1000
    run_SVI(condition_wrapper(measurement=measure_obs), scale_parametrized_guide, guess=guess_obs)
    print(true_mean_std(guess_obs, measure_obs))

    # the performance is actually quite robust to the initial guess for the mean of the posterior
    # it doesn't need to be guess_obs, in this case we set it to 1 and it still works well
    # one criticism is "of course it works well, we've picked the perfect approx-post family"

    guess_obs = 5.
    measure_int = 1000
    run_SVI(intervene_wrapper(measurement=measure_int), scale_parametrized_guide, guess=guess_obs, lr=0.0003, mom=0.9)
    print(5, 1)  # true intervention mean and std

    # the intervention posterior is actually very noisy and doesn't converge well