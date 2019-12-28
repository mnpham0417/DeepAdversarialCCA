import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from network import *

os.makedirs("images", exist_ok=True)

# the size of the new space learned by the model (number of the new features)
outdim_size = 10

# size of the input for view 1 and view 2
input_shape1 = 784
input_shape2 = 784

# number of layers with nodes in each one
layer_sizes1 = [1024, 1024, 1024, outdim_size]
layer_sizes2 = [1024, 1024, 1024, outdim_size]

# the parameters for training the network
learning_rate = 1e-3
epoch_num = 1
batch_size = 800

# the regularization parameter of the network
# seems necessary to avoid the gradient exploding especially when non-saturating activations are used
reg_par = 1e-5

# specifies if all the singular values should get used to calculate the correlation or just the top outdim_size ones
# if one option does not work for a network or dataset, try the other one
use_all_singular_values = False

# if a linear CCA should get applied on the learned features extracted from the networks
# it does not affect the performance on noisy MNIST significantly
apply_linear_cca = True

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)
model_vae_real = VAE()
model_vae_fake = VAE()
model_deepCCA = DeepCCA(layer_sizes1, layer_sizes2, input_shape1,
                    input_shape2, outdim_size, use_all_singular_values)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_VAE_real = optim.Adam(model_vae_real.parameters(), lr=1e-3)
optimizer_VAE_fake = optim.Adam(model_vae_fake.parameters(), lr=1e-3)
optimizer_deepCCA = torch.optim.RMSprop(model_deepCCA.parameters(), lr=learning_rate, weight_decay=reg_par)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader): 

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        # Train VAE real
        # -----------------

        optimizer_VAE_real.zero_grad()
        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model_vae_real(imgs)
        # calculate scalar loss
        loss_vae_real = model_vae_real.loss_function(recon_batch, imgs, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss_vae_real.backward()
        optimizer_VAE_real.step()

        # -----------------
        # Train DeepCCA
        # -----------------

        # Generate a batch of images
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)

        model_deepCCA.zero_grad()

        mu_1, logvar_1 = model_vae_real.encoder(imgs)
        mu_2, logvar_2 = model_vae_fake.encoder(gen_imgs)

        real_encoded = model_vae_real.encoder.reparameterize(mu_1, logvar_1)
        fake_encoded = model_vae_fake.encoder.reparameterize(mu_2, logvar_2)

        decoded1 = [model_vae_real.decoder(z) for z in fake_encoded]
        decoded2 = [model_vae_fake.decoder(z) for z in real_encoded]

        decoded1 = torch.stack(decoded1)
        decoded2 = torch.stack(decoded2)

        view1, view2 = model_deepCCA(decoded1, decoded2)
        print(view1.shape, view2.shape)
        loss_deepCCA = model_deepCCA.loss(view1, view2)
        loss_deepCCA.backward()
        model_deepCCA.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        (g_loss + loss_deepCCA).backward(retain_graph=True)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        # Train VAE fake
        # -----------------
        optimizer_VAE_fake.zero_grad()
        recon_batch_fake, mu_fake, logvar_fake = model_vae_fake(gen_imgs)
        loss_vae_fake = model_vae_fake.loss_function(recon_batch_fake, gen_imgs.detach(), mu_fake, logvar_fake)
        loss_vae_fake.backward()
        optimizer_VAE_fake.step()

        print(
            "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}] [VAE Loss: {}] [VAE Fake Loss: {}] [DeepCCA Loss: {}]".format(epoch, 
            opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), loss_vae_real.item(), loss_vae_fake.item(), loss_deepCCA.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
