import pyro
import pyro.distributions as dist
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate

import torch

# Enumeration is the exact inference method whereby you literally do the integrals
# (sums in the case of discrete variables which are tractable).
# Pyro supports this but it seems to get kind of complex so I'll skip the details
# for now.

def model():
    # Categorical takes (n,) shape tensor where n is num of classes
    z = pyro.sample("z", dist.Categorical(torch.ones(5)))
    print('model z = {}'.format(z))

def guide():
    z = pyro.sample("z", dist.Categorical(torch.ones(5)))
    print('guide z = {}'.format(z))


elbo = Trace_ELBO()
elbo.loss(model, guide)

elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, config_enumerate(guide, "parallel"))
# faster but requires downstream stocfuns to not branch on an outcome

elbo = TraceEnum_ELBO(max_plate_nesting=0)
elbo.loss(model, config_enumerate(guide, "sequential"))