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
from functools import partial
import logging
import os

import tensorflow as tf

from nvidia_tao_tf2.cv.classification.utils.helper import decode_eff as decode_classification_eff
from nvidia_tao_tf2.experimental.decorators.experimental import experimental
from nvidia_tao_tf2.cv.efficientdet.utils.helper import (
    decode_eff as decode_efficientdet_eff,
    load_json_model
)
from nvidia_tao_tf2.cv.efficientdet.utils.keras_utils import restore_ckpt

MB = 1<<20
SUPPORTED_MODELS = ["efficientdet", "classification"]

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
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Architecture of the model to be exported.",
        default=None
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        help="Exponential moving average decay rate if set during training.",
        default=None
    )
    args = vars(parser.parse_args(cl_args))
    return args


def load_efficientdet_model(model_path, key=None, ema_decay=0.998):
    """Load a saved model."""
    if model_path.endswith(".tlt"):
        assert key, "Key should be provided."
        output_model_path, output_model_name = decode_efficientdet_eff(model_path, enc_key=key)
        mode = "eval"
        model = load_json_model(
            os.path.join(output_model_path,
                         f'{mode}_graph.json'))
        restore_ckpt(
            model,
            os.path.join(output_model_path, output_model_name),
            ema_decay=ema_decay,
            steps_per_epoch=0,
            expect_partial=True
        )
        return model
    

def load_classification_model(model_path, key=None):
    """Load a trained classification model."""
    if model_path.endswith(".tlt") and os.path.isfile(model_path):
        assert key, "Key should be provided."
        saved_model = decode_classification_eff(model_path, enc_key=key)
    else:
        assert os.path.isdir(model_path), "Model should be a tf saved model."
        saved_model = model_path
    return saved_model


LOADERS = {
    "classification": load_classification_model,
    "efficientdet": load_efficientdet_model
}

CONVERTERS = {
    "classification": partial(tf.lite.TFLiteConverter.from_saved_model, signature_keys=["serving_default"]),
    "efficientdet": tf.lite.TFLiteConverter.from_keras_model
}
            


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
    mode = args["mode"]
    ema_decay = args["ema_decay"]
    if not output_model_file:
        output_model_file = f"{os.path.splitext(input_model_file)[0]}.tflite"
    # Load a keras or saved model.
    loader_kwargs = {
        "key": key
    }
    if mode == "efficientdet" and not (ema_decay is None):
        loader_kwargs["ema_decay"] = ema_decay
    model = LOADERS[mode](input_model_file, **loader_kwargs)
    logger.info("Converting the saved model to tflite model.")
    converter = CONVERTERS[mode](model)
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
