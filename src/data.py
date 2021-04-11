"""Data loading and processing for the MNIST and CIFAR-100 datasets."""
import gzip
import numpy as np
import pickle
import tarfile

from pathlib import Path
from src.utils import plot_images
from typing import Tuple, List, Union

COMPRESSED_MNIST_NAME = 'mnist.pkl.gz'
COMPRESSED_CIFAR_NAME = 'cifar-100-python.tar.gz'
EXTRACTED_CIFAR_NAME = 'cifar-100-python'

IMAGE_DTYPE = np.float32


def load_mnist(data_dir: Path,
               compressed_file_name: Path = COMPRESSED_MNIST_NAME
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the mnist dataset from .gz file

    Parameters
    ----------
    data_dir : folder within current directory in which the data is saved (default: 'data')
    compressed_file_name : name of compressed mnist file (default: 'mnist.pkl.gz')

    Returns
    -------
    data : a tuple of three datasets, used to construct the training, validation and test sets
    """
    with gzip.open(data_dir / compressed_file_name, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_cifar(data_dir: Path,
               compressed_file_name: Path = COMPRESSED_CIFAR_NAME,
               extracted_dir_name: Path = EXTRACTED_CIFAR_NAME) -> List[np.ndarray]:
    """
    Loads the cifar-100 dataset

    Parameters
    ----------
    data_dir : folder within current directory in which the data is saved (default: 'data')
    compressed_file_name :
    extracted_dir_name :

    Returns
    -------

    """
    extracted_path = data_dir / extracted_dir_name

    # if the extracted folder does not exist in the data directory, extract it
    if not extracted_path.exists():
        tarfile.open(data_dir / compressed_file_name, 'r:gz').extractall(data_dir)

    data = []
    for filename in ['train', 'test']:
        with open(extracted_path / filename, 'rb') as f:
            data.append(pickle.load(f, encoding='bytes'))

    return data


def shuffle(data: Union[Tuple, np.ndarray]):
    """
    Shuffles the data samples

    Parameters
    ----------
    data : un-shuffled input dataset

    Returns
    -------
    data: shuffled data
    """
    if isinstance(data, tuple):  # in case we also want to use labels
        perm_idx = np.random.permutation(len(data[0]))
        for i, d in enumerate(data):
            data[i] = data[perm_idx, :]
    else:
        perm_idx = np.random.permutation(len(data))
        data = data[perm_idx, :]
    return data


def get_dataset(data_name: str,
                train_size: int = 10_000,
                val_size: int = 10_000,
                test_size : int = 10_000,
                shuffle_train_val: bool = True,
                data_dir: Path = Path('data')):
    """
    Loads, processes and returns normalised train, validation and test image sets

    Parameters
    ----------
    data_name : name of dataset ('mnist' or 'cifar-100')
    train_size : size of training set
    val_size : size of validation set
    test_size : size of test set
    shuffle_train_val : shuffle dataset before splitting in training and validation sets
    data_dir : directory to store the data

    Returns
    -------
    x_train: training set array, converted to float 32 to reserve memory
    x_val: validation set array, converted to float 32 to reserve memory
    x_test: test set array, converted to float 32 to reserve memory
    """
    if data_name == 'mnist':
        train_val_data, _, test_data = load_mnist(data_dir)
        x_train_val = train_val_data[0]
        x_test = test_data[0]
    elif data_name == 'cifar-100':
        train_val_dict, test_dict = load_cifar(data_dir)
        x_train_val = train_val_dict[b'data']
        x_test = test_dict[b'data']
    else:
        return ValueError(f'invalid dataset: {data_name}')

    # shuffle data before splitting to training and validation sets
    if shuffle_train_val:
        x_train_val = shuffle(x_train_val)

    # subsample
    if train_size + val_size < len(x_train_val):
        x_train = x_train_val[:train_size, :]
        x_val = x_train_val[train_size:train_size + val_size, :]
    else:
        raise ValueError(f'train ({train_size}) or validation ({val_size}) dataset size too large')

    # subsample
    if test_size < len(x_test):
        x_test = x_test[:test_size, :]
    elif test_size == len(x_test):
        pass
    else:
        raise ValueError(f'test size ({test_size}) too large')

    # scale cifar-100, mnist samples already in [0, 1]
    if data_name == 'cifar-100':
        x_train, x_val, x_test = x_train / 255, x_val / 255, x_test / 255

    return x_train.astype(IMAGE_DTYPE), x_val.astype(IMAGE_DTYPE), x_test.astype(IMAGE_DTYPE)


if __name__ == '__main__':
    x_train_mnist, x_val_mnist, x_test_mnist = get_dataset('mnist')
    x_train_cifar, x_train_cifar, x_train_cifar = get_dataset('cifar-100')

    plot_images(images=x_train_mnist, data_name='mnist', save=False)
    plot_images(images=x_train_cifar, data_name='cifar-100', save=False)
