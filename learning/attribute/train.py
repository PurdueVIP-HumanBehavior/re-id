import torch
from torch.utils import TensorBoard
import argparse


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Training script for attribute model")
    parser.add_argument(
        "--logdir",
        required=True,
        help="Path to logging directory for TensorBoard",
    )
    return parser.parse_args(argv)


def main(argv):
    make_net()
    pass


if __name__ == "__main__":
    sys.exit(main(argv[1:]))
