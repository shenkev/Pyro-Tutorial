import pyro
import pyro.distributions as dist

import torch

# Miscellanous advice
pyro.enable_validation(True)    # <---- This is a good idea for debugging!


# Num, Batch, Events, Shapes
d = dist.Bernoulli(0.5)
# there are 3 shapes we need to worry about when it comes to random variables
# shape = sample_shape + batch_shape + event_shape

# sample_shape: basically how many samples you want to draw from your data
# generating stocfun, e.g.
sample_shape = [2, 2]
print(d.sample(sample_shape))
# draw 9 (3x3 tensor) i.i.d. samples from the distribution

# batch_shape: similar to the notion of "plate", these are the random variables
# that are (conditionally) independent of each other (conditioned) on their
# parents. recall that without conditioning, they are NOT d-separated so this
# point is important.

# event_shape: this describes random variables that are DEPENDENT of each other
# (e.g. multivariate Gaussian has dependent variables where plates don't)

# some examples: single variable
x = d.sample()

assert d.batch_shape == ()
assert d.event_shape == ()
assert x.shape == ()
assert d.log_prob(x).shape == ()

# independent variables

d = dist.Bernoulli(0.5 * torch.ones(3, 4))
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
x = d.sample(sample_shape)
assert x.shape == (2, 2, 3, 4)  # sample_shape comes first, before batach_shape
assert d.log_prob(x).shape == (2, 2, 3, 4)

# basically this is 3*4=12 random variables stacked in a 3x4 matrix, and we pull
# 2*2 = 4 i.i.d. samples of them

# multivariate distribution with dependencies
# here's an example with every shape included
# we use the "expand" function to make replicas of our 3D Gaussian along an
# independent dimension (you can think of them as a Gaussian mixture)

d = dist.MultivariateNormal(torch.zeros(3), torch.eye(3, 3)).expand([10])
assert d.batch_shape == (10,)
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (10, 3)            # == batch_shape + event_shape
# 10 distribution replicas of a 3D Gaussian (dependent amongst dimensions)
x = d.sample(sample_shape)
assert x.shape == (2, 2, 10, 3)            # == batch_shape + event_shape
# 2*2=4 pulls from each of the 10 replica distributions of 3D Gaussians
assert d.log_prob(x).shape == (2, 2, 10)  # == sample_shape + batch_shape

# one confusing thing is you don't know where each shape ends and starts, how do
# we know how a sequence of dimensions (3, 3, 4, 5, 6, 7) corresponds to each
# of sample_shape, batch_shape, and event_shape?


# weirdly you can create (dependent) joint distributions from univariate
# distributions using the "to_event" function

d = dist.Bernoulli(0.5 * torch.ones(3,4,5)).to_event(1)
# only 1 dimension (from the right) should be considered dependent
assert d.batch_shape == (3,4)
assert d.event_shape == (5,)
x = d.sample()
assert x.shape == (3, 4, 5)
assert d.log_prob(x).shape == (3,4)

# btw what IS a dependent multivariate Bernoulli distribution?
# https://stats.stackexchange.com/questions/7089/probability-formula-for-a-multivariate-bernoulli-distribution


# Nested "Plates"
# plate only converts the right-most (batch) dimension to independence
# if you have a 2D image for example, you need to appy plate twice

# with pyro.plate("x_axis", 320):
#     # within this context, batch dimension -1 is independent
#     with pyro.plate("y_axis", 200):
#         # within this context, batch dimensions -2 and -1 are independent


# The end of this section describes some more advanced methods for parallelizing
# or optimizing the computation. It's probably not necessary for now.