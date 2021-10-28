# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

from iva.common.export.keras_exporter import KerasExporter as Exporter
from iva.common.types.base_ds_config import BaseDSConfig

logger = logging.getLogger(__name__)


class ClassificationExporter(Exporter):
    """Define an exporter for classification models."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 backend="uff",
                 classmap_file=None,
                 **kwargs):
        """Initialize the classification exporter.

        Args:
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.

        Returns:
            None.
        """
        super(ClassificationExporter, self).__init__(model_path=model_path,
                                                     key=key,
                                                     data_type=data_type,
                                                     strict_type=strict_type,
                                                     backend=backend)
        self.classmap_file = classmap_file

    def set_input_output_node_names(self):
        """Set input output node names."""
        self.output_node_names = ["predictions/Softmax"]
        self.input_node_names = ["input_1"]

    def load_classmap_file(self):
        """Load the classmap json."""
        data = None
        with open(self.classmap_file, "r") as cmap_file:
            try:
                data = json.load(cmap_file)
            except json.decoder.JSONDecodeError as e:
                print(f"Loading the {self.classmap_file} failed with error\n{e}")
                sys.exit(-1)
            except Exception as e:
                if e.output is not None:
                    print(f"Classification exporter failed with error {e.output}")
                sys.exit(-1)
        return data

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        if not os.path.exists(self.classmap_file):
            raise FileNotFoundError(
                f"Classmap json file not found: {self.classmap_file}")
        data = self.load_classmap_file()
        if not data:
            return []
        labels = [""] * len(list(data.keys()))
        if not all([class_index < len(labels)
                    and isinstance(class_index, int)
                    for class_index in data.values()]):
            raise RuntimeError(
                "Invalid data in the json file. The class index must "
                "be < number of classes and an integer value.")
        for class_name, class_index in data.items():
            labels[class_index] = class_name
        return labels

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        channel_index = 0 if self.data_format == "channels_first" else -1
        if input_dims[channel_index] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 1 for classification
            "network_type": 1
        }
        if num_classes:
            kwargs["num_classes"] = num_classes
        if self.backend == "uff":
            kwargs.update({
                "input_names": self.input_node_names,
                "output_names": self.output_node_names
            })

        ds_config = BaseDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            **kwargs
        )
        return ds_config
