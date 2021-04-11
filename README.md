# Kernel Density Estimation
Density Estimation of MNIST and CIFAR-100 using Gaussian Kernel Density Estimation (Numpy only). Running time is improved by utilising Numba, a just-in-time compiler for Python which compiles a subset of the language into efficient machine code that is comparable in performance to a traditional compiled language. It also enables execution of loops in parallel on separate threads (using Numba’s `prange` instead of `range`).

## Download data

After git cloning the repository, download the image datasets:

```shell
$ mkdir data/
$ curl https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz --output cifar-100-python.tar.gz
$ curl http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz --output mnist.pkl.gz
```

## Installation

Create a conda environment and install dependencies:
```shell
conda env create -f environment.yml
conda activate kde
```
