"""
Computes the mean log-probability of MNIST / CIFAR-100 vaidation / test sets using the Gaussian KDE method

Example usage:
1. For a quick demonstration on our approach (on 10 test samples), run:
python -m src.run --data-name mnist --test --demo

2. To reproduce our result for the MNIST test set for one seed (0) and optimal sigma (0.2), run:
python -m src.run --data-name mnist --test

3. To reproduce our result for the CIFAR-100 test set for one seed (0) and optimal sigma (0.2), run:
python -m src.run --data-name cifar-100 --test

4. To reproduce our result for multiple seeds, run:
python -m src.run --data-name mnist --test --seeds 0 10 20 30 40
"""
import datetime
import numpy as np
import logging
import time

from arguments import ArgumentParser
from collections import defaultdict
from src.data import get_dataset
from src.kde import kde_gauss
from src.utils import config_logging, save_json


if __name__ == '__main__':

    # initialise argument parser and logger
    args = ArgumentParser().parse_args()
    suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    log_name = f'{suffix}_{args.data_name}_test_{len(args.seeds)}seeds'
    config_logging(save=args.save_logs, log_name=log_name)

    # log arguments
    logging.info('COMMAND LINE ARGS:')
    logging.info(args)

    stats = defaultdict(list)
    for seed in args.seeds:
        np.random.seed(seed)

        # get training and validation set; random seed influences the training and validation sets
        if args.demo:
            # limit data for a quick demonstration
            x_train, x_val, x_test = get_dataset(data_name=args.data_name, train_size=10, val_size=10, test_size=10,
                                                 data_dir=args.data_dir)
        else:
            # sample 10k by default
            x_train, x_val, x_test = get_dataset(data_name=args.data_name, data_dir=args.data_dir)

        # GET THE MEAN LOG-PROBABILITY FOR X-VAL / X-TEST
        start_time = time.time()
        mean_log_likelihood = kde_gauss(data_a=x_train, data_b=x_test if args.test else x_val, sigma=args.sigma)
        end_time = time.time() - start_time

        # log stats
        format_str = 'Data: {} | Seed: {:.0f} | Mean log-likelihood : {: .2f} | Time: {: .2f}'
        logging.info(format_str.format(*[args.data_name, seed, mean_log_likelihood, end_time]))

        stats['time'].append(end_time)
        stats['mean_log_likelihood'].append(mean_log_likelihood)

    # average stats across seeds
    test_results = {key: (np.average(stat), np.std(stat)) for key, stat in stats.items()}

    # if more than one seed, also output the averaged results
    if len(args.seeds) > 1:
        format_str = 'Data: {} | Mean log-likelihood Avg: {: .2f} | Mean log-likelihood Std: {: .2f} | ' \
                     'Time Avg: {: .2f} | Time Std: {: .2f}'
        logging.info(format_str.format(*[args.data_name, test_results['mean_log_likelihood'][0],
                                         test_results['mean_log_likelihood'][1], test_results['time'][0],
                                         test_results['time'][1]]))

    if args.save_results:
        save_json(data=test_results, file_name=log_name)
