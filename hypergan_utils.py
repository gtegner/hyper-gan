import torch.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.distributions as D
import scipy


class Mixture:
    def __init__(self, mix, comp, z_dim, num_classes=3):

        self.mix = mix
        self.comp = comp
        self.z_dim = z_dim
        self.num_classes = num_classes

    def sample_z_mixture(self, batch_size, z_dim=None):
        if z_dim is None:
            z_dim = self.z_dim

        mix_sample = self.mix.sample((batch_size,))
        comp_sample = self.comp.sample((batch_size,))
        mask = F.one_hot(
            mix_sample, num_classes=self.num_classes).unsqueeze(-1)
        mask_r = mask.repeat(1, 1, self.z_dim).float()
        samples_mixed = mask_r * comp_sample
        samples_ = torch.sum(samples_mixed, 1)
        return samples_


def true_data(N=1000, range=6):
    x = np.linspace(-range, range, N)
    y = x**3
    return torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).float().unsqueeze(1)


def plot_inference(x, y, X_train, y_train, inference, save_path=None, args=None):
    st = inference.detach().cpu().numpy().std(0).squeeze()
    mu = inference.detach().cpu().numpy().mean(0).squeeze()

    kwargs = {'alpha': 0.3, 'color': 'black'}
    hi_ci = mu + 3 * st
    low_ci = mu - 3 * st

    x = x.cpu()
    y = y.cpu()
    X_train = X_train.cpu()
    y_train = y_train.cpu()

    plt.figure()
    plt.plot(x[:, 0].numpy(), y[:, 0].numpy(), label='True data')
    plt.scatter(X_train[:, 0].numpy(), y_train[:,
                                               0].numpy(), label='Training data')
    plt.plot(x[:, 0].numpy(), mu, label='Mean')
    plt.fill_between(x[:, 0].numpy(), hi_ci, low_ci, **kwargs)
    plt.ylim((-100, 100))
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    if args.show_plot:
        plt.show()


def plot_classification(mainNetwork, X_train, y_train, figure_name, args):
    points = generate_points(X_train, y_train, N=11)

    # Forward pass on points
    points_torch = torch.from_numpy(points).float()
    if args.device == 'cuda':
        points_torch = points_torch.cuda()

    print("Begin forward pass")
    print(points.shape)

    predictions = mainNetwork(points_torch)
    mu = predictions.mean(0)
    mu = torch.softmax(mu, dim=1)

    entropy = get_entropy(mu)
    fig, ax = plt.subplots()
    fig, ax = plot_contour(points, entropy, "Entropy", ax=ax, fig=fig)
    fig.savefig(figure_name)

    return None


def get_entropy(mean):
    entropies = []
    for row in mean:
        entropy = scipy.stats.entropy(row.cpu().detach())
        entropies.append(entropy)
    return np.array(entropies)


def plot_contour(points, entropy, title, levels=10, ax=None, fig=None):
    N = np.int_(np.sqrt(points.shape[0]))
    entropy = entropy.reshape((N, N))
    x = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), N)
    y = np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), N)

    if ax is None or fig is None:
        plt.figure()
        cs = plt.contourf(x, y, entropy, levels=levels, cmap='Blues')
        cbar = plt.colorbar(cs)
        plt.suptitle(title)
    else:
        cs = ax.contourf(x, y, entropy, levels=levels, cmap='Blues')
        fig.colorbar(cs, ax=ax)
        ax.title.set_text(title)
    return fig, ax


def sample_z(batch_size, z_dim, mu=0, sigma=1, device='cpu', prior='uniform'):
    if prior == 'uniform':
        return torch.rand((batch_size, z_dim)).to(device) * 2 - 1
    elif prior == 'normal':
        return mu + sigma * torch.randn((batch_size, z_dim)).to(device)
    else:
        raise NotImplementedError


def generate_random_networks(n, z_dim, mainNet, generator, args):
    z = sample_z(n, z_dim, device=args.device, prior=args.prior)

    if args.experiment == 'mutual_information' or args.experiment == 'mi_condition_on_z':
        random_conditional = z if args.experiment == 'mi_condition_on_z' else generator.sample_conditional(
            z.shape[0], args.conditional_dim)
        weights, biases, mixed, _ = generator(z, random_conditional)

    if args.experiment == 'discriminator':
        weights, biases, mixed = generator(z)

    mainNet.update_weights(weights, biases)
    return mainNet, mixed


def generate_mixture(num_classes, z_dim, network_layers, latent_dim):
    num_classes = 3
    mix = D.Categorical((torch.ones(num_classes,)))
    comp = D.Normal(torch.rand(num_classes, z_dim).float(),
                    torch.ones(num_classes, z_dim)*0.1)

    mix = Mixture(mix, comp, z_dim, num_classes)

    num_classes = 3
    full_latent_dim = latent_dim * (len(network_layers)-1)
    prior_mix = D.Categorical((torch.ones(num_classes,)))
    prior_comp = D.Normal(torch.rand(num_classes, full_latent_dim).float(
    ), torch.ones(num_classes, full_latent_dim)*0.1)
    prior_mixture = Mixture(prior_mix, prior_comp,
                            full_latent_dim, num_classes)
    return prior_mixture


def generate_points(x_train, y_train, N=100, r=10):
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()

    x_range = (x_train[:, 0].min()-r, x_train[:, 0].max()+r)
    y_range = (x_train[:, 1].min()-r, x_train[:, 1].max()+r)

    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    coord = np.array([[(i, j) for i in x] for j in y]).reshape(-1, 2)
    return coord
