import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from typing import Dict, List


def check_results_path(results_dir_name: str = 'results'):
    """
    Creates a results folder if it does not exist in the current directory

    Returns
    ----------
    results_path : path of results directory
    """
    results_path = Path(results_dir_name)
    if not results_path.exists():
        Path.mkdir(results_path)
    return results_path


def config_logging(save: str = False, log_name: str = 'log', log_dir: str = 'logs'):
    """
    Configures the logger

    Parameters
    ----------
    save : whether to save the logger's output
    log_name : name of logging file
    log_dir : directory of logging files
    """
    if save:
        log_path = Path(log_dir)

        # if the log directory does not exist, create a folder!
        if not log_path.exists():
            Path.mkdir(log_path)

        file_path = Path(log_path) / Path(log_name + '.log')
        handlers = [logging.FileHandler(filename=file_path),
                    logging.StreamHandler(sys.stdout)]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        handlers=handlers
    )


def save_json(data, file_name: str = 'hsweep'):
    """
    Saves experimental results in a json file

    Parameters
    ----------
    data : experimental results
    file_name : name of results file
    """
    # if results directory does not exist, create it!
    results_path = check_results_path()

    file_path = results_path / Path(file_name + '.json')

    with open(file_path, 'w') as f:
        json.dump(data, f)


def plot_images(images: np.ndarray, data_name: str, grid_height: int = 8, grid_width: int = 8, save: bool = False):
    """
    Plots a grid of MNIST and CIFAR-100 images

    Parameters
    ----------
    images : image dataset
    data_name : name of dataset (mnist or cifar-100)
    grid_height : height of image grid to plot
    grid_width : width of image grid to plot
    save : saves the plotted figure
    """
    # reshape images to original image dimensions
    if data_name == 'mnist':
        img_height, img_width = (28, 28)
        images = images.reshape((-1, img_height, img_width))
        image_grid = np.zeros((grid_height * img_height, grid_width * img_width))
    elif data_name == 'cifar-100':
        img_height, img_width = (32, 32)
        images = images.reshape((-1, 3, img_height, img_width))
        images = np.transpose(images, (0, 2, 3, 1))

        image_grid = np.zeros((grid_height * img_height, grid_width * img_width, 3))
    else:
        return ValueError(f'invalid dataset: {data_name}')

    # random subset of images to plot (in case data not shuffled already, we can see possibly see a more diverse set)
    plot_indices = np.random.randint(low=0, high=len(images), size=grid_height * grid_width)

    # fill a grid of shape grid_height x grid_width with randomly sampled images
    for i in range(grid_height):
        for j in range(grid_width):
            plot_idx = plot_indices[i * grid_width + j]
            image_grid[i * img_height:(i + 1) * img_height, j * img_width:(j + 1) * img_width] = images[plot_idx]

    plt.imshow(image_grid, cmap='gray')
    plt.axis('off')

    if save:
        # if results directory does not exist, create it!
        results_path = check_results_path()
        plt.savefig(results_path / Path(f'{data_name}_sample.png'))

    plt.show()


def plot_sweep(sigma_grid: List[float], logl_sigmas: List[float], data_name: str, save: bool = False,
               plot_name: str = 'plot'):
    """
    Plots hyperaparameter sweep results: sigma vs log-probability, with variability across seeds

    Parameters
    ----------
    sigma_grid : standard deviation values used in the hyperparameter sweep
    logl_sigmas : mean log-probability values per sigma (averaged across seeds if multiple seeds are used)
    data_name : name of dataset ('mnist' or 'cifar-100')
    save : whether to save the plot or not
    plot_name : name of the plot
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()

    plt.plot(sigma_grid, logl_sigmas['mean'], linewidth=0.7)
    ax.fill_between(sigma_grid, logl_sigmas['min'], logl_sigmas['max'], alpha=.2, linewidth=0.0)

    plt.title(f'{data_name} | $\sigma$ vs L$_{{D_{{B}}}}$')
    plt.ylabel('mean log-likelihood (L$_{D_B})$')
    plt.xlabel('standard deviation ($\sigma$)')

    if save:
        # if results directory does not exist, create it!
        results_path = check_results_path()
        plt.savefig(results_path / Path(plot_name + '.png'))

    plt.show()


def log_stats(stats: Dict[str, List], data_list: List[float]):
    """
    Computes and stores hyperparameter sweep stats across seeds for one sigma

    Parameters
    ----------
    stats : stores statistics across seeds for each sigma; each value on a list corresponds to one sigma,
            aggregated across seeds. Structure: {'mean': [], 'std': [], 'min': [], 'max': []}
    data_list : list of values for every seed, for one sigma (length = number of seeds)

    Returns
    -------
    stats : updated dictionary with new mean, std, min and max values across seeds
    """
    avg = np.average(data_list)
    std = np.std(data_list)

    stats['mean'].append(avg)
    stats['std'].append(std)
    stats['min'].append(np.min(data_list))
    stats['max'].append(np.max(data_list))
    return stats, avg, std