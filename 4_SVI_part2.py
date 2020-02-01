import pyro
import pyro.distributions as dist


# This section introduces the notion of exploiting conditional independence between
# variables to run VI in minibatches.
# To allow Pyro to use minibatches, we need to tell it that the observational vars
# obs_i are independent. We draw on the "plate" idea in graphical model, indicating
# the obs_i's are actually replicates of each other, independent and generated given
# "latent_fairness".

def model(data):
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))
    # loop over the observed data [WE ONLY CHANGE THE NEXT LINE]
    for i in pyro.plate("data_loop", len(data)):
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


# tensorized version
def model_tensor(data):
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))

    with plate('observe_data'):
        pyro.sample('obs', dist.Bernoulli(f), obs=data)


# automatic (random) subsampling
# the idea is the .step() and .evaluate_loss() functions of the SVI estimator will
# use minibatches, much like how neural network trains in minibatches
# the loss at each time step will be scaled by dataset_size/minibatch_size
# the scaling is why we need to tell plate both the dataset size and minibatch size

with plate('observe_data', size=10, subsample_size=5) as ind:
    pyro.sample('obs', dist.Bernoulli(f),
                obs=data.index_select(0, ind))


# one problem is plate by default uses a random sampling, instead you maay want to
# make sure you iterate over the entire dataset, even if the  minibatches are random
# see the docs for customer subsampling strategies.


# This tutorial gets a bit more complicated. There are a couple of other major ideas
# (which seem too advanced to be relevant). One is the notion of thinking about
# nontrivial cases where independence exists between variables. The second is using
# "plate" in complex ways, by putting it into the "guide", by recursing it, etc.