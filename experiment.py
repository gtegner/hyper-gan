from itertools import product
import subprocess
import os

from tqdm import tqdm

tasks = ['regression']
toy_sigmas = [9, 16]

epochs = [1000]
lrs = [1e-4, 2e-4]
betas = [25.0, 50.0]

experiments = ['discriminator', 'mutual_information', 'mi_condition_on_z']
fit_lambdas = [1e-1]
weight_batches = [1, 32]


mi_lrs = [1e-2, 1e-3, 2e-3]


def str_product(*args):
    for k in product(*args):
        yield [str(v) for v in k]


def num_product(*args):
    p = 1
    for k in args:
        p *= len(k)
    return p


args = [toy_sigmas, tasks, weight_batches, epochs, lrs,
        betas, experiments, fit_lambdas, mi_lrs]

pbar = tqdm(total=num_product(*args))

print("Training a total of {} tasks".format(num_product(*args)))

for toy_sigma, task, weight_batch, epoch, lr, beta, experiment, fit_lambda, mi_lr in tqdm(str_product(*args)):
    pbar.set_description(
        "Processing " + ', '.join([weight_batch, epoch, lr, beta, experiment, fit_lambda, mi_lr]))

    if (experiment != 'discriminator' and weight_batch == '1') or (experiment == 'discriminator' and weight_batch != '1'):
        pbar.update(1)
        continue

    subprocess.call(['python', 'main.py',
                     '--toy_sigma', toy_sigma,
                     '--dataset', task,
                     '--weight_batch', weight_batch,
                     '--n_iter',
                     epoch,
                     '--lr',
                     lr,
                     '--beta', beta, '--experiment', experiment, '--fit_lambda', fit_lambda, '--mi_lr', mi_lr, '--show_plot', 'false'])  # , stdout=FNULL, stderr=subprocess.STDOUT)
    pbar.update(1)
