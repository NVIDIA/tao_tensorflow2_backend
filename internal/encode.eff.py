import argparse
import os
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback
from cv.efficientdet.utils.helper import encode_eff

def parse_command_line(args):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='EFF Decode Tool')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to the saved_model.')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str,
                        help='The path to tlt model.')
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        default=None,
                        type=str,
                        help='encryption key.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_command_line(args)
    encode_eff(args.model, args.output, args.key)
    print("Decode successfully.")


if __name__ == "__main__":
    main()
