# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Export a classification model."""

# import build_command_line_parser as this is needed by entrypoint
from iva.common.export.app import build_command_line_parser as global_parser # noqa pylint: disable=W0611
from iva.common.export.app import run_export
from iva.makenet.export.classification_exporter import ClassificationExporter as Exporter


def build_command_line_parser(parser=None):
    """Simple function to build the command line parser."""
    args_parser = global_parser(parser=parser)
    args_parser.add_argument(
        "--classmap_json",
        help="UNIX path to classmap.json file generated during classification <train>",
        default=None,
        type=str,
    )
    return args_parser


def parse_command_line(args=None):
    """Parse command line arguments."""
    parser = build_command_line_parser(parser=None)
    return vars(parser.parse_known_args(args)[0])


def main(args=None):
    """Run export for classification."""
    args = parse_command_line(args=args)
    run_export(Exporter, args=args)


if __name__ == "__main__":
    main()
