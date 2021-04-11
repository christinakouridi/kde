import argparse
from pathlib import Path


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()

        self.add_argument("--data-name", type=str, default='mnist',
                          help="name of dataset to train on, possible values: mnist, cifar-100")
        self.add_argument("--data-dir", type=Path, default='data',
                          help="data directory")
        self.add_argument("--demo", action='store_true',
                          help="whether to limit the dataset to 10 samples for a quick demonstration of our approach")
        self.add_argument("--save-results", action='store_true',
                          help="whether to store the experimental results. creates a 'results' folder in the current "
                               "directory by default")
        self.add_argument("--save-logs", action='store_true',
                          help="whether to store the output of the logger. creates a 'logs folder in the current "
                               "directory by default")
        self.add_argument("--seeds", type=int, default=[0], nargs='+',
                          help="seeds for random processes")
        self.add_argument("--sigma", type=float, default=0.2,
                          help="scale parameter of Gaussian Kernel")
        self.add_argument("--test", action='store_true',
                          help="whether to compute the mean log-probability of the test set. Evaluation is on the "
                               "validation set by default")

    def parse_args(self, args=None):
        """
        Parse the arguments and perform some basic validation
        """
        args = super().parse_args(args)
        return args




