# TAO Toolkit TensorFlow2

Home for the Deep Learning components and solutions for TAO Toolkit with TensorFlow2.

## Steps to run the prototype text classification example

As soon as the repository is cloned, run the envsetup file to update build enviroments as follows

```sh
source scripts/envsetup.sh
```

We recommend adding this command to your local `~/.bashrc` file, so that every new terminal instance receives this.

### 1. Run the docker container

The TLT tensorflow-2 base environment docker has already been built and uploaded on gitlab for the developers. For instantiating the docker, simply run the `tao_tf.py` script. The usage for the container is mentioned below.

```sh
usage: tao_tf [-h] [--gpus GPUS] [--volume VOLUME] [--env ENV]
              [--mounts_file MOUNTS_FILE] [--shm_size SHM_SIZE]
              [--run_as_user] [--tag TAG] [--ulimit ULIMIT]

Tool to run the pytorch container.

optional arguments:
  -h, --help            show this help message and exit
  --gpus GPUS           Comma separated GPU indices to be exposed to the docker.
  --volume VOLUME       Volumes to bind.
  --env ENV             Environment variables to bind.
  --mounts_file MOUNTS_FILE
                        Path to the mounts file.
  --shm_size SHM_SIZE   Shared memory size for docker
  --run_as_user         Flag to run as user
  --tag TAG             The tag value for the local dev docker.
  --ulimit ULIMIT       Docker ulimits for the host machine.

```

A sample command to instantiate the docker is mentioned below.

```sh
tao_tf -- 
```

### 2. Download example data and run training

Once you are able to download/instantiate the docker, you may launch the script you wish to run by simply appending the command and its args to the end of the `tao_tf` command. For example, to launch the classification train entrypoint, you may use the sample command below.

```sh
tao_tf --volume /path/to/output/results_dir:/path/to/output/results_dir -- python cv/classification/scripts/train.py
```

Please note that the `tao_tf` command requires a `--` separator to separate the args for the `tao_tf` command and the script to be run in the docker and it's respective args.

### 3. Updating the base docker

There will be situations where developers would be required to update the third party dependencies to newer versions, or upgrade CUDA etc. In such a case, please follow the steps below:

#### 1. Build base docker

The base dev docker is defined in `$NV_TAO_TF2_TOP/docker/Dockerfile`. The python packages required for the TLT dev is defined in `$NV_TAO_TF2_TOP/docker/requirements.txt`. Once you have made the required change, please update the base docker using the build script in the same directory.

```sh
cd $NV_TAO_TF2_TOP/docker
./build_and_run.sh --build
```

### 2. Test the newly built base docker

The build script tags the newly built base docker with the username of the account in the user's local machine. Therefore, the developers may tests their new docker by using the `tao_tf` command with the `--tag` option.

```sh
tao_tf --tag $USER -- script args
```

For example, to run classification, you may use:

```sh
tao_tf --tag $USER -- python cv/classification/scripts/train.py --help
```

### 3. Update the new docker

Once you are sufficiently confident about the newly built base docker, please do the following

1. Push the newly built base docker to the registry

    ```sh
    bash $NV_TAO_TF2_TOP/docker/build.sh --build --push
    ```

2. The above step produces a digest file associated with the docker. This is a unique identifier for the docker. So please note this, and update all references of the old digest in the repository with the new digest. You may find the old digest in the `$NV_TAO_TF2_TOP/docker/manifest.json`.

Push you final updated changes to the repository so that other developers can leverage and sync with the new dev environment.

Please note that if for some reason you would like to force build the docker without using a cache from the previous docker, you may do so by using the `--force` option.

```sh
bash $NV_TAO_TF2_TOP/docker/build.sh --build --push --force
```
