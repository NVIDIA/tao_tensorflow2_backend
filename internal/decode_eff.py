import argparse
import os
import zipfile

from eff.core import Archive, File
from eff.callbacks import BinaryContentCallback


def decode_eff(eff_model_path, output_path=None, passphrase=None):
    """Decode EFF to saved_model directory.

    Args:
        eff_model_path (str): Path to eff model
        passphrase (str, optional): Encryption key. Defaults to None.

    Returns:
        str: Path to the saved_model
    """
    # Decrypt EFF
    eff_filename = os.path.basename(eff_model_path)
    eff_art = Archive.restore_artifact(
        restore_path=eff_model_path,
        artifact_name=eff_filename,
        passphrase=passphrase)
    zip_path = eff_art.get_handle()
    # Unzip
    ckpt_path = output_path or os.path.dirname(zip_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    # TODO(@yuw): try catch? 
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(ckpt_path)
    return ckpt_path


def parse_command_line(args):
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='EFF Decode Tool')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        help='Path to the EFF file.')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        type=str,
                        help='The path to output directory.')
    parser.add_argument('-k',
                        '--key',
                        required=False,
                        default=None,
                        type=str,
                        help='encryption key.')
    return parser.parse_args(args)


def main(args=None):
    args = parse_command_line(args)
    decode_eff(args.model, args.output, args.key)
    print("Decode successfully.")


if __name__ == "__main__":
    main()