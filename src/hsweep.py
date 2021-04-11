""""
It finds the optimal value of the standard deviation hyperparameter (sigma) for MNIST, and CIFAR-100. The optimal sigma
for each dataset corresponds to the maximum log-likelihood across sigmas, where the log-likehood is first
averaged across seeds

Example usage:
1. For a quick demonstration (on 10 CIFAR-100 validation samples), run:
python -m src.hsweep --seeds 0 10 --demo --data-name cifar-100

2. To reproduce our results for MNIST for one seed (0), run:
python -m src.hsweep

3. To reproduce our results for MNIST for 5 seeds, run:
python -m src.hsweep --seeds 0 10 20 30 40
"""

import datetime
import logging
import numpy as np

from arguments import ArgumentParser
from collections import defaultdict
from src.data import get_dataset
from src.kde import kde_gauss
from src.utils import config_logging, log_stats, plot_sweep, save_json


SIGMA_GRID = [0.05, 0.08, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]


if __name__ == '__main__':

    # initialise argument parser and logger
    args = ArgumentParser().parse_args()
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

    log_name = f'{suffix}_hsweep_{args.data_name}_{len(args.seeds)}seeds'
    config_logging(save=args.save_logs, log_name=log_name)

    # stores optimal hyperparameters for each dataset based on the optimal average mean log-likelihood across seeds
    results = {}

    # stores statistics across seeds for each sigma; each value on a list corresponds to one sigma,
    # aggregated across seeds. Structure: {'mean': [], 'std': [], 'min': [], 'max': []}
    logl_sigmas = defaultdict(list)

    for sigma in SIGMA_GRID:

        # stores the log likelihood for each seed for this sigma
        logl_seeds = []

        for seed in args.seeds:
            np.random.seed(seed)

            # get training and validation set; random seed influences the training and validation sets
            if args.demo:
                # limit data for a quick demonstration
                x_train, x_val, _ = get_dataset(data_name=args.data_name, train_size=10, val_size=10, test_size=10,
                                                data_dir=args.data_dir)
            else:
                # sample: 10k
                x_train, x_val, _ = get_dataset(data_name=args.data_name, data_dir=args.data_dir)

            # GET THE MEAN LOG-PROBABILITY FOR X-VAL
            logl = kde_gauss(data_a=x_train, data_b=x_val, sigma=sigma)

            # store mean log-likelihood for each seed
            logl_seeds.append(logl)

        # aggregate results across seeds, store them in the logl_sigmas dict, and log them
        logl_sigmas, logl_seeds_avg, logl_seeds_std = log_stats(logl_sigmas, logl_seeds)
        format_str = 'Data: {} | Sigma: {:.2f} | Log-Likelihood Avg: {: .2f} | Log-likelihood Std: {: .2f}'
        logging.info(format_str.format(*[args.data_name, sigma, logl_seeds_avg, logl_seeds_std]))

    # idx of mean log-likelihood with the highest average across seeds
    idx_max = np.argmax(logl_sigmas['mean'])

    # store optimal values
    opt_sigma = SIGMA_GRID[idx_max]
    opt_logl_mean = logl_sigmas['mean'][idx_max]
    opt_logl_std = logl_sigmas['std'][idx_max]
    results[args.data_name] = (opt_sigma, opt_logl_mean, opt_logl_std)

    # log optimal results with indication of variance across seeds
    format_str = 'Data: {} | Optimal Sigma: {:.2f} | Max Log-Likelihood Avg: {: .2f} | ' \
                 'Max Log-likelihood Std: {: .2f}'
    logging.info(format_str.format(*[args.data_name, opt_sigma, opt_logl_mean, opt_logl_std]))

    # plot results
    plot_sweep(SIGMA_GRID, logl_sigmas, args.data_name, plot_name=log_name, save=args.save_results)

    # saves optimal results
    if args.save_results:
        save_json(data=results, file_name=log_name)