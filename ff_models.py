import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
from mine.models.mine import Mine
from mine.models.layers import ConcatLayer, CustomSequential


class GenLayer(nn.Module):
    def __init__(self, input_dim, layer_dim, bias_dim):
        super().__init__()
        h = 8
        self.layer_dim = layer_dim
        self.layers = nn.Sequential(nn.Linear(input_dim, layer_dim + bias_dim))
        # nn.ReLU(),
        # nn.Linear(h, layer_dim + bias_dim))

    def forward(self, x):
        x = self.layers(x)
        layer_out = x[:, :self.layer_dim]
        bias_out = x[:, self.layer_dim:]

        return layer_out, bias_out


class Generator(nn.Module):
    def __init__(self,
                 z_dim,
                 latent_dim,
                 architecture,
                 args,
                 beta,
                 bias=True,
                 mi_estimator=None,
                 opt_mi=None):
        super().__init__()

        self.args = args
        h = 512
        self.n_layers = len(architecture) - 1

        mixer_dim = self.n_layers * latent_dim

        self.mi_estimator = mi_estimator

        self.mixer = nn.Sequential(nn.Linear(z_dim, h), nn.ReLU(), nn.Linear(
            h, h), nn.ReLU(), nn.Linear(h, mixer_dim))

        self.generators = self.init_generator(
            latent_dim, architecture)

        self.architecture = architecture
        self.beta = beta

        self.opt_mi = opt_mi

    @property
    def method(self):
        return self.args.experiment

    def init_generator(self, latent_dim, architecture):
        gen_layers = []
        for i in range(len(architecture) - 1):
            gen_layers.append(
                GenLayer(latent_dim, architecture[i] * architecture[i+1], architecture[i+1]))

        return nn.ModuleList(gen_layers)

    def forward(self, z, conditional=None):
        if conditional is not None:
            z = torch.cat((z, conditional), 1)

        batch_size = z.size(0)
        mixed = self.mixer(z)

        mi = 0
        if self.mi_estimator is not None and conditional is not None:
            mi = self.mi_estimator(mixed, conditional)

        mixed_split = torch.chunk(
            mixed, self.n_layers, dim=1)  # batch x latent

        weight_layers = []
        biases = []

        for i, (layer) in enumerate(self.generators):
            head = mixed_split[i]

            weight, bias = layer(head)

            weight = weight.view(
                batch_size, self.architecture[i+1], self.architecture[i])

            weight_layers.append(weight.clone())
            biases.append(bias.clone())

        if conditional is None:
            return weight_layers, biases, mixed

        elif conditional is not None:
            return weight_layers, biases, mixed, mi

    def sample_conditional(self, batch_size, num_labels):
        random_labels = torch.randint(
            low=0, high=num_labels, size=(batch_size, ))
        one_hot = torch.zeros((batch_size, num_labels))
        one_hot[torch.arange(batch_size), random_labels] = 1.0

        if self.args.cuda:
            one_hot = one_hot.cuda()
        return one_hot

    def train_step(self, X_train_batch,
                   y_train_batch,
                   z_batch,
                   opt_g,
                   opt_d,
                   mainNet,
                   loss_fn,
                   bce_loss=None,
                   discriminator=None,
                   high_entropy_mixture=None,
                   conditional=None):

        is_classification = False
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            is_classification = True

        opt_g.zero_grad()
        if self.opt_mi is not None:
            self.opt_mi.zero_grad()
        if opt_d is not None:
            opt_d.zero_grad()

        if self.method == 'mutual_information':
            conditional = self.sample_conditional(
                z_batch.size(0), self.args.conditional_dim)
        elif self.method == 'mi_condition_on_z':
            conditional = z_batch

        if self.method == 'discriminator':
            weights, biases, mixed = self.forward(z_batch, conditional)

        elif self.method == 'mutual_information' or self.method == 'mi_condition_on_z':
            weights, biases, mixed, mi = self.forward(z_batch, conditional)

        mainNet.update_weights(weights, biases)

        if self.method == 'discriminator':
            if high_entropy_mixture is None:
                high_entropy_dist = torch.randn_like(
                    mixed).to(self.args.device)
            else:
                high_entropy_dist = high_entropy_mixture.sample_z_mixture(
                    mixed.size(0))
            disc_real = discriminator(high_entropy_dist)

            real_labels = torch.ones((mixed.size(0), 1)).to(self.args.device)
            fake_labels = torch.zeros((mixed.size(0), 1)).to(self.args.device)

            gen_adv_loss = bce_loss(discriminator(mixed), real_labels)

        elif self.method == 'mutual_information' or self.method == 'mi_condition_on_z':
            gen_adv_loss = mi

        # Update Generator
        out = mainNet(X_train_batch)  # weight_batch x batch_size x y_dim
        net_loss = 0

        acc = 0

        for outj in out:
            net_loss += loss_fn(outj, y_train_batch)
            if is_classification:
                acc += torch.argmax(outj, 1).eq(y_train_batch).float().mean()

        net_loss = net_loss / len(out)
        if is_classification:
            acc = acc / len(out)

        gen_loss = self.args.fit_lambda * net_loss + self.beta * gen_adv_loss
        gen_loss.backward()
        opt_g.step()
        if self.opt_mi is not None:
            self.opt_mi.step()

        # Update discriminator
        if self.method == 'discriminator':
            disc_loss = 0.5 * (bce_loss(disc_real, real_labels)
                               + bce_loss(discriminator(mixed.detach()), fake_labels))
            disc_loss.backward()
            opt_d.step()
            to_return = {
                'net_loss': net_loss.item(),
                'gen_adv_loss': gen_adv_loss.item(),
                'disc_loss': disc_loss.item()
            }

        elif self.method == 'mutual_information' or self.method == 'mi_condition_on_z':
            to_return = {
                'net_loss': net_loss.item(),
                'gen_adv_loss': gen_adv_loss.item(),
            }

        if is_classification:
            to_return.update({'accuracy': acc})

        return to_return


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        h = 128
        self.layers = nn.Sequential(
            nn.Linear(input_dim, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, 1), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class MainNetwork(nn.Module):
    def __init__(self, weight_layers=None, biases=None, activation='relu'):
        super().__init__()

        self.weight_layers = weight_layers
        self.biases = biases

        self.activation = activation

        if activation == 'relu':
            self.activation = lambda x: F.relu(x)
        elif activation == 'tanh':
            self.activation = lambda x: F.tanh(x)

    def weight_stats(self):
        layer_stats = dict()
        mean_std = 0
        for ix, layer in enumerate(self.weight_layers):
            layer_np = layer.cpu().detach().numpy()
            l1 = np.linalg.norm(layer_np, axis=(1, 2))

            mu = l1.mean()
            std = l1.std()

            diversion = std / mu

            mean_std += diversion

            layer_stats['relative_std_' +
                        str(ix)] = float(diversion)

        mean_std /= len(self.weight_layers)
        layer_stats['total_relative_std'] = float(mean_std)

        return layer_stats

    def forward(self, x):

        num_layers = len(self.weight_layers)
        batch_size = self.weight_layers[0].shape[0]  # Number of ensembles

        passes = []
        for i in range(batch_size):  # Iterate over batch
            x_hat = x.clone()
            for j in range(num_layers):
                wi = self.weight_layers[j][i]
                bi = self.biases[j][i]
                x_hat = F.linear(x_hat, wi, bias=bi)
                if j != num_layers - 1:
                    x_hat = self.activation(x_hat)
            passes.append(x_hat.clone())

        passes = torch.stack(passes, 0)
        return passes

    def update_weights(self, weights, biases):
        self.weight_layers = weights
        self.biases = biases
