# Table of contents
  * [Mounting network shares](#mounting-network-shares)
    * [Dataset mounts](#dataset-mounts)
    * [Workspace mount](#workspace-mount)
    * [Docker mounts](#docker-mounts)
    * [Shared workspace setup](#shared-workspace-setup)
  * [Installation](#installation)
  * [Training a 20-class car make model](#training-a-20-class-model)
    * [Using the shell script in makenet](#run-with-shell-script)
    * [Using the moduluspy launcher in local environment](#run-with-maglev-sdk)
  * [Evaluating a trained model](#evaluating-a-trained-model)
  * [Visualizing kernel norms of a pruned model](#visualizing-kernels)
  * [Exporting a trained model to Caffe or UFF format](#exporting-a-trained-model-to-caffe-or-uff-format)
  * [Generating INT8 calibration table for a trained model](#generating-int8-calibration-table)
  * [Launching a job in cosmos](#running-with-sdk-launcher)
    * [Usage instructions](#usage-instructions)
    * [Cosmos setup](#cosmos-setup)
      * [Using classification runner for Cosmos launching](#using-makenet-runner-for-cosmos-launching)

## Mounting network shares

Most of the IVA datasets are located on shared drives. Please follow instructions mentioned below to mount these drives.

### Dataset mounts

At the very least, you will need to mount `projects2_metropolis`. Sample mount command:

```sh
sudo mkdir /media/projects2_metropolis
sudo mount -t cifs -o "vers=2.0,username=<your_username>,dom=nvidia.com,dir_mode=0777,file_mode=0777" //dc2-dgx-netapp2/projects2_metropolis /media/projects2_metropolis
ln -s /media/projects2_metropolis /home/projects2_metropolis

sudo mkdir /media/IVAData2
sudo mount -t cifs -o "rw,vers=1.0,username=vpraveen,dom=nvidia.com,dir_mode=0777,file_mode=0777" //netapp-hq11/IVAData2 /media/IVAData2
ln -s /media/IVAData2 /home/IVAData2
```
Replace <your_username> with your nvidia unix account username.

Then, make sure to create the following symlink:

```sh
ln -s /home/projects2_metropolis/datasets/maglev_tfrecords ~/datasets
```

If you would like to run the dlav test cases, you will need to follow the share mount drives instructions provided in dlav/drivenet.
For permissions to access the drive, please submit a request to join the DL at https://dlrequest/
As your scratch space, please create a directory in `/home/projects2_metropolis/tmp/$USER`, where `$USER` is your nvidia username.

## Installation

See https://ai-infra.gitlab-master-pages.nvidia.com/ai-infra/installation.html#installation
for instructions on cloning the ai-infra repository as well as setting up your environment.
We support only the Dazel environment.

Make sure that your datasets are available in `~/datasets`.

For all IVA datasets, the tfrecords for the datasets are available in the projects2_metropolis shared drive.

For all your experiment results, please create your scratch space in `/home/projects2_metropolis/tmp/${USER}/experiments`.

## Training a 20-class car make model

The iva/makenet/scripts/train.py runs the basic training pipeline for classification tasks. You may invoke this script by running the following command.

```sh
dazel run //iva/makenet/scripts:train -- <command_line_args_for_train_script>
```
The basic command line arguments are as follows:
1. --train_root
                      Path to spec file. Absolute path or relative to
                      working directory. If not specified, default spec from
                      spec_loader.py is used.
2. --val_root
                      Path to a folder where experiment outputs should be
                      written.
3. --epochs
                      Absolute path to the model file used for initializing
                      model weights.
4. --batch_size
                      Name of the model file. If not given, then defaults to
                      model.hdf5.
5. --arch {resnet, vgg}
                      Model architecture to be constructed.
6. --nlayers
                      number of layers to be constructed. For ResNet models, choose from [10, 18, 34, 50]
                      For VGG models, choose from [16, 19]

This is a simple single GPU training session. Inorder to invoke a multiGPU training session there are two ways to do so:
1. Using a shell script in makenet
2. Using the moduluspy launcher in local environment.

### Using the shell script in makenet

You may use the `train_mutligpu.sh` script in `/iva/makenet/scripts` to run a multiGPU training session using the following command.

```sh
dazel run //iva/makenet/scripts:train_multigpu -- np <num_GPUs> <command_line_args_for_train_script>
```

Sample command line to run a multiGPU training session for the carmake secondary detector.


```sh
dazel run //iva/makenet/scripts:train_multigpu -- -np 4 --train_root /home/projects2_metropolis/datasets/secondary/make_reduced/dataset_6k/split/train --val_root /home/projects2_metropolis/datasets/secondary/make_reduced/dataset_6k/split/val --epochs 80 --batch_size 64 --learning_rate 0.02 --arch resnet --nlayers 18 --output_dir /home/projects2_metropolis/tmp/$USER/make6k_exp/resnet18_make6k_b64_lr2e-2_pretrained_l1 --step_size 10 --pretrained_model /home/projects2_metropolis/tmp/$USER/$YOUR_PRETRAINED_MODEL
```

### Using the moduluspy launcher in local environment

You may also use the generic maglev sdk launcher to run a multiGPU training session using the following command.

## Launching a job in cosmos

Launching a job in cosmos is possible using the SDK launcher available in moduluspy. NOTE: This feature has not been moved to the master branch. The sample format for a command line is.

```sh
dazel run //iva/common:run -- LAUNCHER_ARGS -- SCRIPT_NAME SCRIPTS_ARGS
```
For example, building on the last command line, in order to run a sample training job on cosmos with 4 GPUs on cluster id 425, you may use the following command.

Please replace $USER with you unix id.
