# TFLite convertor

This tool helps convert a Keras model trained by the TAO Toolkit TF1/TF2 container to a TFLite model.

Follow the instructions [here](../../../README.md#instantiating-the-development-container) to get instantiate the dev container and run the converter.

## <a name='Runningtheconverter'></a>Running the converter

The sample usage for the converter

```sh
usage: export_tflite [-h] [--model_file MODEL_FILE] [--key KEY] [--output_file OUTPUT_FILE] --mode {efficientdet,classification} [--ema_decay EMA_DECAY]

Export keras models to tflite.

options:
  -h, --help            show this help message and exit
  --model_file MODEL_FILE
                        Path to a model file.
  --key KEY             Key to load the model.
  --output_file OUTPUT_FILE
                        Path to the output model file.
  --mode {efficientdet,classification}
                        Architecture of the model to be exported.
  --ema_decay EMA_DECAY
                        Exponential moving average decay rate if set during training.
```

Sample command to run the tflite converter.

```sh
python export_tflite.py --model_file /path/to/model.[tlt/hdf5] \
                        --output_file /path/to/model.tflite \
                        --key $KEY
                        --mode classification
```

Output from a successful conversion run.

```txt
WARNING: This is an experimental module. Please use this at your own risk.
INFO: Model provided is a saved model directory at ${model_path}
INFO: Converting the saved model to tflite model.
INFO: Writing out the tflite model.
TFLite model of size 34 MB was written to ${output_model_path}
```
