import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from hypergan_utils import sample_z, true_data, plot_inference, generate_random_networks, plot_classification
from ff_models import Generator, MainNetwork, Discriminator
from toy_dataset import load_toy_dataset, generate_classification_data

import argparse
import os

import matplotlib.pyplot as plt
from typing import List

from collections import defaultdict

from mine.models.mine import Mine
import copy

MODEL_DIR = 'SAVED_MODELS'
FIGURES_DIR = 'FIGURES'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(FIGURES_DIR):
    os.mkdir(FIGURES_DIR)


def save_model(model, file_name):
    torch.save(model, os.path.join(MODEL_DIR, file_name + '.pt'))


def encode(args):
    encoding = str(abs(hash(str(args))) % 10**8)
    return encoding


def model_name_from_args(args, encoded=True):
    encoding = encode(args)
    return f'{args.dataset}-{args.experiment}_iters_{args.n_iter}_weight_batch_{args.weight_batch}_beta_{args.beta}_conditional_dim_{args.conditional_dim}_lr_{args.lr}_zprior_{args.prior}_latent_{args.latent_dim}_{encoding}_{args.toy_sigma}'


def get_loss_from_inference(yhat, y_test, loss_fn):
    avg_loss = 0
    std_loss = 0
    for out in yhat:
        loss = loss_fn(out, y_test)
        avg_loss += loss.item()
        std_loss += loss.item()**2

    avg_loss /= yhat.shape[0]
    std_loss = std_loss / yhat.shape[0] - avg_loss**2

    return avg_loss, std_loss


def get_ensemble_loss_from_inference(yhat, y_test, loss_fn):

    ensemble_preds = torch.mean(yhat, 0)
    loss = loss_fn(ensemble_preds, y_test)

    return loss.item()


def test(X_test, y_test, mainNet, generator, loss_fn, args, num_nets=1000):

    if args.device == 'cuda':
        X_test = X_test.cuda()
        y_test = y_test.cuda()

    to_return = {}

    if not isinstance(num_nets, list):
        num_nets = [num_nets]

    for num_net in num_nets:
        yhat = inference(X_test, y_test, args.z_dim, mainNet, generator,
                         args, num_nets=num_net)

        avg_loss, std_loss = get_loss_from_inference(yhat, y_test, loss_fn)
        ensemble_loss = get_ensemble_loss_from_inference(yhat, y_test, loss_fn)

        to_return.update({
            f'loss_{num_net}': avg_loss,
            f'loss_std_{num_net}': std_loss,
            f'ensemble_loss_{num_net}': ensemble_loss
        })

    return to_return


def train(data, opt_g, opt_d, mainNet, generator, discriminator, loss_fn, bce_loss, args):
    X_train, y_train = data.get('x_train'), data.get('y_train')
    X_test, y_test = data.get('x_test'), data.get('y_test')

    if args.device == 'cuda':
        X_train = X_train.cuda()
        y_train = y_train.cuda()

        X_test = X_test.cuda()
        y_test = y_test.cuda()

    for epoch in range(args.n_iter):
        z_batch = sample_z(args.weight_batch, args.z_dim,
                           device=args.device, prior=args.prior)

        for i in range(len(X_train) // args.batch_size):
            X_train_batch = X_train[i * args.batch_size: (i+1)*args.batch_size]
            y_train_batch = y_train[i * args.batch_size: (i+1)*args.batch_size]

            if args.experiment == 'discriminator':
                train_dict = generator.train_step(X_train_batch, y_train_batch, z_batch,
                                                  opt_g, opt_d,
                                                  mainNet, loss_fn, bce_loss, discriminator=discriminator, high_entropy_mixture=None)

            elif args.experiment == 'mutual_information' or args.experiment == 'mi_condition_on_z':
                train_dict = generator.train_step(X_train_batch, y_train_batch, z_batch,
                                                  opt_g, opt_d,
                                                  mainNet, loss_fn, bce_loss, discriminator=discriminator, high_entropy_mixture=None)

        if epoch % args.log_interval == 0:
            if args.experiment == 'discriminator':
                to_print = "Epoch {}, Loss: {}, Generator: {}, Discriminator: {}".format(
                    epoch, train_dict.get('net_loss'), train_dict.get('gen_adv_loss'), train_dict.get('disc_loss'))
            elif args.experiment == 'mutual_information' or args.experiment == 'mi_condition_on_z':
                to_print = "Epoch {}, Loss: {}, MI: {}".format(
                    epoch, train_dict.get('net_loss'), train_dict.get('gen_adv_loss'))
            if args.dataset == 'classification':
                to_print += " Accuracy {}".format(train_dict.get('accuracy'))

            print(to_print)

            num_net = 100
            random_networks, _ = generate_random_networks(
                num_net, args.z_dim, mainNet, generator, args)
            yhat = random_networks(X_test)

            avg_loss, std_loss = get_loss_from_inference(yhat, y_test, loss_fn)
            ensemble_loss = get_ensemble_loss_from_inference(
                yhat, y_test, loss_fn)

            weight_stats = random_networks.weight_stats()

            inference_results = {
                f'loss_{num_net}': avg_loss,
                f'loss_std_{num_net}': std_loss,
                f'ensemble_loss_{num_net}': ensemble_loss
            }

            save_stats(weight_stats, inference_results, epoch=epoch, args=args)

    random_networks, _ = generate_random_networks(
        100, args.z_dim, mainNet, generator, args)
    weight_stats = random_networks.weight_stats()

    inference_results = test(data.get('x_test'), data.get('y_test'), mainNet, generator,
                             loss_fn, args, num_nets=[1, 5, 10, 100])

    save_stats(weight_stats, inference_results, epoch=epoch, args=args)

    if args.save_model:
        save_model(generator, model_name_from_args(args))

    if args.plot:
        X_plot, y_plot = true_data(range=6)
        if args.device == 'cuda':
            X_plot = X_plot.cuda()
            y_plot = y_plot.cuda()
        plot_result(X_train, y_train, X_plot, y_plot, args.z_dim,
                    random_networks, generator, args)
        # plot_mixed(X_train, y_train, args.z_dim, mainNet, generator, args)

    print("Training finished")


def save_stats(*stats, epoch, args):
    encoding = {'id': encode(args)}
    args_dict = copy.deepcopy(vars(args))

    args_dict.update(encoding)
    args_dict.update({'epoch': epoch})

    for stat in stats:
        args_dict.update(stat)

    columns = list(args_dict.keys())
    df = pd.DataFrame(columns=[str(x) for x in columns])
    df = df.append(args_dict, ignore_index=True)

    if os.path.exists('logs.csv'):
        logs = pd.read_csv('logs.csv', sep='|')
        new_df = logs.append(df, ignore_index=True, sort=False)
        new_df.to_csv('logs.csv', sep='|', index=False)
    else:
        df.to_csv('logs.csv', sep='|', index=False)
        # logs = pd.read_csv('logs.csv', sep='|')


def inference(x, y, z_dim, mainNet, generator, args, num_nets=1000):
    if args.device == 'cuda':
        x = x.cuda()
        y = y.cuda()

    mainNet, _ = generate_random_networks(
        num_nets, z_dim, mainNet, generator, args)

    yhat = mainNet(x)
    return yhat


def plot_result(X_train, y_train, X_test, y_test, z_dim, mainNet, generator, args):
    yhat = inference(X_test, y_test, z_dim, mainNet, generator, args)
    save_path = f'{FIGURES_DIR}/{model_name_from_args(args)}.png'

    if args.dataset == 'regression':
        plot_inference(X_test, y_test, X_train, y_train, yhat, save_path, args)
    elif args.dataset == 'classification':
        plot_classification(mainNet, X_train, y_train, save_path, args)


def main(args):

    if args.dataset == 'regression':
        x_train, _, y_train, _ = load_toy_dataset(20, sigma=args.toy_sigma)
        x_test, y_test = true_data(range=4)
    if args.dataset == 'classification':
        N = 2000
        sigma = 5.0
        x_train, y_train, x_test, y_test, _, _ = generate_classification_data(
            N, sigma)

        y_train, y_test = y_train.squeeze(-1), y_test.squeeze(-1)

    data = {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

    h = 100

    x_dim = x_train.shape[-1]

    if args.dataset == 'regression':
        out_dim = 1
    elif args.dataset == 'classification':
        out_dim = 3

    # 2 layer net
    network_layers = [x_dim, h, h, out_dim]

    mi_estimator = None
    opt_mi = None
    if args.experiment == 'mutual_information' or args.experiment == 'mi_condition_on_z':
        assert args.conditional_dim is not None, "Must specify label dimensions"

        z_dim = args.z_dim
        mixer_dim = (len(network_layers) - 1) * args.latent_dim

        conditional_dim = z_dim if args.experiment == 'mi_condition_on_z' else args.conditional_dim

        # Initalize mutual-information estimator
        mi_model = nn.Sequential(nn.Linear(mixer_dim + conditional_dim, 400), nn.ReLU(
        ), nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 1))

        mi_estimator = Mine(mi_model, loss='mine_biased', method='concat')

        # Adjust input dimension to include labels
        z_dim = z_dim + conditional_dim
        opt_mi = torch.optim.Adam(mi_estimator.parameters(), lr=args.mi_lr)
    else:
        z_dim = args.z_dim

    generator = Generator(z_dim, args.latent_dim, architecture=network_layers,
                          beta=args.beta, args=args, mi_estimator=mi_estimator, opt_mi=opt_mi).to(device)
    mainNet = MainNetwork(activation='relu').to(device)

    if args.experiment == 'discriminator':
        discriminator = Discriminator(
            (len(network_layers)-1) * args.latent_dim).to(device)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
        bce_loss = nn.BCELoss()
    elif args.experiment == 'mutual_information' or args.experiment == 'mi_condition_on_z':
        discriminator = None
        opt_d = None
        bce_loss = None

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr)

    if args.dataset == 'regression':
        loss_fn = nn.MSELoss()
    elif args.dataset == 'classification':
        loss_fn = nn.CrossEntropyLoss()

    train(data, opt_g, opt_d, mainNet,
          generator, discriminator, loss_fn, bce_loss, args)


def str2bool(value):
    return value.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='regression',
                        help="Regression or Classification")
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--weight_batch', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--experiment', type=str, default='mutual_information',
                        help="Choose between discriminator, mutual_information, mi_condition_on_z")
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--save_model', type=str2bool, default=False)
    parser.add_argument('--conditional_dim', type=int, default=10)
    parser.add_argument('--plot', type=str2bool, default=True)
    parser.add_argument('--prior', type=str, default='uniform')
    parser.add_argument('--fit_lambda', type=float, default=1e-1)
    parser.add_argument('--mi_lr', type=float, default=1e-2)
    parser.add_argument('--show_plot', type=str2bool, default=False)
    parser.add_argument('--toy_sigma', type=float, default=9)

    args = parser.parse_args()

    args.cuda = (device == 'cuda') and (args.use_cuda == True)
    args.device = device

    main(args)
