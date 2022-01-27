# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Script to prune the classification TLT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import tempfile
import tensorflow as tf

from cv.makenet.utils.helper import encode_eff
from cv.makenet.pruner.pruner import ClassificationPruner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logger = logging.getLogger(__name__)


def build_command_line_parser(parser=None):
    """Build a command line parser for pruning."""
    if parser is None:
        parser = argparse.ArgumentParser(description="TAO TF2 pruning script")
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        help="Path to the target model for pruning",
                        required=True,
                        default=None)
    parser.add_argument("-o",
                        "--output_file",
                        type=str,
                        help="Output file path for pruned model",
                        required=True,
                        default=None)
    parser.add_argument("--results_dir",
                        type=str,
                        default=None,
                        help="Path to where the status log is generated.")
    parser.add_argument('-k',
                        '--key',
                        required=True,
                        type=str,
                        help='Key to load a .tlt model')
    parser.add_argument('-n',
                        '--normalizer',
                        type=str,
                        default='max',
                        help="`max` to normalize by dividing each norm by the \
                        maximum norm within a layer; `L2` to normalize by \
                        dividing by the L2 norm of the vector comprising all \
                        kernel norms. (default: `max`)")
    parser.add_argument('-eq',
                        '--equalization_criterion',
                        type=str,
                        default='union',
                        help="Criteria to equalize the stats of inputs to an \
                        element wise op layer. Options are \
                        [arithmetic_mean, geometric_mean, union, \
                        intersection]. (default: `union`)")
    parser.add_argument("-pg",
                        "--pruning_granularity",
                        type=int,
                        help="Pruning granularity: number of filters to remove \
                        at a time. (default:8)",
                        default=8)
    parser.add_argument("-pth",
                        "--pruning_threshold",
                        type=float,
                        help="Threshold to compare normalized norm against \
                        (default:0.1)", default=0.1)
    parser.add_argument("-nf",
                        "--min_num_filters",
                        type=int,
                        help="Minimum number of filters to keep per layer. \
                        (default:16)", default=16)
    parser.add_argument("-el",
                        "--excluded_layers", action='store',
                        type=str, nargs='*',
                        help="List of excluded_layers. Examples: -i item1 \
                        item2", default=[])
    parser.add_argument("-v",
                        "--verbose",
                        action='store_true',
                        help="Include this flag in command line invocation for \
                        verbose logs.")
    return parser


def parse_command_line_arguments(args=None):
    """Parse command line arguments for pruning."""
    parser = build_command_line_parser()
    return parser.parse_args(args)


def run_pruning(args=None):
    """Prune an encrypted Keras model."""
    # Set up logger verbosity.
    verbosity = 'INFO'
    if args.verbose:
        verbosity = 'DEBUG'
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=verbosity
    )
    # results_dir = args.results_dir
    # if results_dir is not None:
    #     if not os.path.exists(results_dir):
    #         os.makedirs(results_dir)

    assert args.equalization_criterion in \
        ['arithmetic_mean', 'geometric_mean', 'union', 'intersection'], \
        "Equalization criterion are [arithmetic_mean, geometric_mean, union, \
         intersection]."
    assert args.normalizer in ['L2', 'max'], \
        "normalizer options are [L2, max]."

    pruner = ClassificationPruner(args.model, args.key)

    # Pruning trained model
    pruned_model = pruner.prune(
        threshold=args.pruning_threshold,
        excluded_layers=args.excluded_layers)

    # Save the encrypted pruned model
    tmp_saved_model = tempfile.mkdtemp()
    pruned_model.save(tmp_saved_model)
    encode_eff(tmp_saved_model, args.output_file, args.key)

def main(args=None):
    """Wrapper function for pruning."""
    # parse command line
    args = parse_command_line_arguments(args)
    run_pruning(args)


if __name__ == "__main__":
    main()