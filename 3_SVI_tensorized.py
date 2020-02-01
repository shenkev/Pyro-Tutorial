import math
from tqdm import tqdm
import torch
from torch.distributions import constraints

import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist


# Reimplementation of 3_SVI.py with tensors

def model(alpha0=10.0, beta0=10.0):
    
    alpha = torch.tensor(alpha0)
    beta = torch.tensor(beta0)
    
    def model_inner(data):
    
        f = pyro.sample("z_fairness", dist.Beta(alpha, beta))
        f_tensor = f*torch.ones(data.shape)
        pyro.sample("x", dist.Bernoulli(f_tensor), obs=data)
        # THE MAIN DIFFERENCE is data and f_tensor are tensors of size [len(data),]
        # rather than scalar values
    
    return model_inner

def guide(data):

    a = pyro.param("alpha_q", torch.tensor(15.0), constraint=constraints.positive)
    b = pyro.param("beta_q", torch.tensor(15.0), constraint=constraints.positive)

    pyro.sample("z_fairness", dist.Beta(a, b))

pyro.clear_param_store()

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)

svi = SVI(model(), guide, optimizer, loss=Trace_ELBO())

NPos = 10
NNeg = 5
data = torch.tensor(NPos*[1.0] + NNeg*[0.0])

n_steps = 2000
for step in tqdm(range(n_steps)):
    svi.step(data)

alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()


inferred_mean = alpha_q / (alpha_q + beta_q)
factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
inferred_std = inferred_mean * math.sqrt(factor)

print("\nbased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))
print("True posterior based on counting real and pseudoflips: {}".format((NPos+10.0)/(NPos+NNeg+20.0)))