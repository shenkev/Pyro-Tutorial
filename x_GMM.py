import os
from collections import defaultdict
import torch
import numpy as np
import scipy.stats
from torch.distributions import constraints
from matplotlib import pyplot
# %matplotlib inline

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete


pyro.enable_validation(True)

# Gaussian Mixture Model
# consider a 1D Gaussian model, based on the data below, clearly we need
# a mixture of (2) Gaussians to fit it well

data = torch.tensor([0., 1., 10., 11., 12.])

