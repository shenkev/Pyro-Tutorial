import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs")

from tqdm import tqdm
import numpy as np
import torch
import torchvision.datasets as dset
import torch.nn as nn
import torchvision.transforms as transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# I think the trickiest part of this tutorial is realizing that we need to define a separate z_i
# for each sample x_i because each z_i has a unique prior/posterior.
# We really cannot represent the VAE properly if we only have two variables, one for z and one
# for x. The setting is much better represented using "plates" or replications of the variables.
# In Pyro, the z_i's and x_i's are called local variables, those that are different for each
# data samples. The mu_i's and sigma_i's are local parameters.

# A second tricky point is the plates have dimension (mini)batch_size rather than the entire
# dataset size. This is because new random variables are being constructed in each batch. We're
# doing optimization of the dynamics P(X|Z) at the same time we're doing inference q(Z|X). In
# each minibatch, we're improving our inference q(Zb|Xb) on a particular batch of data Xb.
# Hopefully there's transfer between the data samples, s.t. improving the inference on one
# batch helps the infrence on another batch - that's the idea of tying the dynamics of the
# samples together with a neural network.
# At the same time, in each minibatch, given the current inference q(Zb|Xb), we try to improve
# the dynamics P(X|Z) to the "true dynamics" given the posterior is correct.


# for loading and batching MNIST dataset
def setup_data_loaders(batch_size=128, use_cuda=False):
    root = './data'
    download = True
    trans = transforms.ToTensor()
    train_set = dset.MNIST(root=root, train=True, transform=trans,
                           download=download)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    kwargs = {'num_workers': 1, 'pin_memory': use_cuda}
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
        batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader

# prediction model
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        self.softplus = nn.Softplus()  # soft relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):

        hidden = self.softplus(self.fc1(z))
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

# (latent) inference model
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, x):

        x = x.reshape(-1, 784)
        hidden = self.softplus(self.fc1(x))

        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class VAE(nn.Module):

    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    # x: is batch_size X 784 size
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):  # we need to identify a separate z for each x, each z_i has a unique prior/posterior
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))  # unit Gaussian prior, constant values
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))  # prior distribution
            # to_event makes sure this is a multivariate normal instead of many univariate ones

            # decode the latent code z
            loc_img = self.decoder.forward(z)  # forward dynamics
            # score against actual images
            pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))  # observational distribution
            # to_event because the neural networks ties the pixels together
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):  # we need to identify a separate z for each x, each z_i has a unique prior/posterior
            z_loc, z_scale = self.encoder.forward(x)
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))  # approximate posterior, why are the dims dependent? the covariate matrix is diagonal
            # because the neural networks tie them together

    def reconstruct_img(self, x):
        z_loc, z_scale = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        loc_img = self.decoder(z)
        return loc_img


vae = VAE().to(device)
optimizer = Adam({"lr": 1.0e-3})
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
train_loader, test_loader = setup_data_loaders()


def train(train_loader):

    epoch_loss = 0

    for batch, _ in tqdm(train_loader):
        batch = batch.to(device)
        epoch_loss += svi.step(batch)

    return epoch_loss/len(train_loader.dataset)

def plot_vae_samples(vae, epoch):
    x = torch.zeros([1, 784]).to(device)
    for i in range(10):
        sample_loc_i = vae.model(x)
        img = sample_loc_i[0].view(1, 28, 28).cpu().data.numpy()
        writer.add_image('image', img, epoch)

def evaluate(test_loader, epoch):

    test_loss = 0.

    for i, (batch, _) in enumerate(tqdm(test_loader)):
        batch = batch.to(device)
        test_loss += svi.evaluate_loss(batch)

        if i == 0:
            plot_vae_samples(vae, epoch)

    return test_loss / len(test_loader.dataset)


num_epochs = 50
pyro.clear_param_store()
train_elbo = []
test_elbo = []

for epoch in range(num_epochs):

    total_epoch_loss_train = train(train_loader)
    train_elbo.append(-total_epoch_loss_train)
    writer.add_scalar('ELBO/train', -total_epoch_loss_train, epoch)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

    if epoch % 2 == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(test_loader, epoch)
        test_elbo.append(-total_epoch_loss_test)
        writer.add_scalar('ELBO/test', -total_epoch_loss_test, epoch)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))