import torch
import pyro

# The basic unit of probabilistic programming is the stochastic function, it's composed of:
# 1. deterministic function (e.g. function, nn.Module)
# 2. primitive stochastic functions that calls a random number generator
# Pyro calls stochastic functions "models"
# It seems like stofuncs are data generating functions or probability distributions
# (represented) in their generation form. Exact relationship will be seem in a bit.

pyro.set_rng_seed(101)  # make things reproducible, especially since Pyro uses RNGs

# Primitive Stofuncs
# Ok so indeed the tutorial calls stofuncs distributions. But it's still interesting to note
# representational differences of distributions in math and in code (as stofuncs).
# Pyro actually piggybacks off Torch's distribution library which has all your usual
# exponential-family distributions implemented. These are primitive stofuncs.
# Of course this is a bit limiting, so you can define arbitrary probability distributions
# by transforming the primitive stofuncs. But doing so most likely leads to approximiations
# when you want these abstract distributions to realize some values.
# e.g. Torch's Transform class allows:
#            invertable transformations with computable log det jacobians
# there are a bunch of other transformation functions defined, I wonder if a generic
# neural network is also allowed in the library, most likely (if outputsize==inputsize)
# Pyro defines a pyro.distributions library that is a light wrapper around torch.distributions

normal = torch.distributions.Normal(loc=0, scale=1)  # unit normal
samp = normal.rsample()
print("sample", samp)  # drawing a sample is... cheap? depends on approximation method
print("log prob", normal.log_prob(samp))  # evaluating the density is almost always cheap
# main difference between .rsample() and .sample() seems to be computing gradients


# A Simple StoFunc Example
# As seen below, a stofunc is a probability distribution (in this a case a joint), describing
# how samples are drawn (returned) from the data generating process. This is enough for the
# Pyro engine to do other things with the distribution (I think?)
# e.g. if you had reflection on this code, you can compute the density of a particular point
# based on the parameters of the Bernoulli and Normal distributions
# It's a toy example of the DAG: cloud_cover --> temperature

def weather():
    cloudy = torch.distributions.Bernoulli(0.3).sample()
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 15.0, 'sunny': 25.0}[cloudy]
    scale_temp = {'cloudy': 5.0, 'sunny': 8.0}[cloudy]
    temp = torch.distributions.Normal(mean_temp, scale_temp).rsample()
    return cloudy, temp.item()

# To summarize: define a stofunc by the data generating function of the distribution
# Pyro does the heavy lifting and does inference over it (I think)


# Ok to work with distributions in Pyro, you basically just switch to the pyro.distribution
# libray and most function calls are the same as the torch.distribution library
pyro_samp = pyro.sample("sampling instance 1", pyro.distributions.Normal(0, 1))
# you need to give the sample a name in the first parameter because Pyro's backend needs to
# uniquely identify computations to perform inference

def pyro_weather():
    cloudy = pyro.sample("cloudy sampling 1", pyro.distributions.Bernoulli(0.3))
    cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
    mean_temp = {'cloudy': 15.0, 'sunny': 25.0}[cloudy]
    scale_temp = {'cloudy': 5.0, 'sunny': 8.0}[cloudy]
    temp = pyro.sample("temp sampling 1", pyro.distributions.Normal(mean_temp, scale_temp))
    return cloudy, temp.item()

# pyro_weather() is a STOCHASTIC FUNCTION, a basic unit of probabilistic programming
# it's stochastic because the returned values are not deterministic
# it's a function, obviously, that returns samples
# it implicitly defines a joint distribution over two random variables: cloudy, temperature
# to understand why it's useful to work with data generator functions and implicit
# distributions, realize a complex distribution can be implicitly defined from a simple one
# by simply putting the generated samples through a transformation.
# closed-form definition of complex distributions would be very hard to define symbolically.
# remark: Pyro refers to joint distribution, stofunc, and probabilistic model interchangeably


# Here is an example of defining a complex distrib P(sales), we don't even have to think
# about the closed form solution.
# Note that ice_cream_sales is another stofunc, an atom of the Pyro library.
# The graph is now cloudy --> temp --> sales, cloudy --> sales
def ice_cream_sales():
    cloudy, temp = pyro_weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_cream', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream


# How powerful is Pyro's inference engine?
# it works on arbitrarily complex deterministic Python functions, which is pretty damn powerful
# here is an example where you can work with functions that even have recursion
# the geometric distribution defines P(N=n), the chance of getting the first head on the Nth try

def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)

# this again, is a data-generating function, it returns the number of tries required to get
# the first head and given this, Pyro can perform inference which is pretty cool
# btw, it's not even Pytorch, it's python code
# note however, you had to get each sampling instance a different name, the reason is not clear
# to me at this point

print(geometric(0.2))


# You can also define higher order distributions by creating functions that return stofuncs
# in the example below, construct_normal is a distribution over normal distributions
# not clear to me at this point whether construct_normal --> normal_product is the correct
# graph interpretation of this

def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y

# this function gives us insight into why we need to name sampling instances, they define the
# variables! e.g. z1 and z2 are two variables (e.g. in a Directed Graph)
# if Pyro's backend is doing graph manipulation, they need names for the nodes!

def construct_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn

# construct_normal returns a stofunc, it is not a stofunc in the sense it doesn't
# return samples from a value-distribution
# however, it is a higher-order stofunc in the sense it returns samples from
# distribution-of-distributions. Pyro considers it a stofunc as well
# construct_normal returns a distribution with a chosen mean, and a user speficied variance

print(construct_normal()(1.))

# Final conclusion: Pyro is universal
# i.e. can be used to represent any computable probability distribution