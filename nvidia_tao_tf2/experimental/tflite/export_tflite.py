# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os

import tensorflow as tf

from nvidia_tao_tf2.cv.classification.utils.helper import decode_eff
from nvidia_tao_tf2.experimental.decorators.experimental import experimental

MB = 1<<20

logger = logging.getLogger(__name__)


def parse_command_line(cl_args="None"):
    """Parse command line args."""
    parser = argparse.ArgumentParser(
        prog="export_tflite",
        description="Export keras models to tflite."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Path to a model file."
    )
    parser.add_argument(
        "--key",
        type=str,
        default="",
        help="Key to load the model."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Path to the output model file."
    )
    args = vars(parser.parse_args(cl_args))
    return args


@experimental
def main(cl_args=None):
    """Main wrapper to run the tflite converter."""
    # Convert the model
    args = parse_command_line(cl_args=cl_args)
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [TAO Toolkit] [%(levelname)s] %(name)s %(lineno)d: %(message)s',
        level="INFO"
    )
    input_model_file = args["model_file"]
    output_model_file = args["output_file"]
    key = args["key"]
    if not output_model_file:
        output_model_file = f"{os.path.splitext(input_model_file)[0]}.tflite"

    if os.path.isdir(input_model_file):
        logger.info(
            "Model provided is a saved model directory at {}".format(
                input_model_file
            )
        )
        saved_model = input_model_file
    else:
        saved_model = decode_eff(
            input_model_file,
            enc_key=key
        )
    logger.info("Converting the saved model to tflite model.")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model,
        signature_keys=["serving_default"],
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()
    model_root = os.path.dirname(output_model_file)
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    logger.info("Writing out the tflite model.")
    with open(output_model_file, "wb") as tflite_file:
        model_size = tflite_file.write(tflite_model)

    print(f"TFLite model of size {model_size//MB} MB was written to {output_model_file}")

if __name__ == "__main__":
    main()
