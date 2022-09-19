ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.07-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy requirement and install
WORKDIR /workspace/
COPY docker/requirements.txt /workspace/
RUN pip install nvidia-pyindex
RUN pip install -r requirements.txt && \
    rm -rf requirements.txt

# install public version of Nvidia QAT Toolkit
RUN git clone https://github.com/NVIDIA/TensorRT.git
COPY docker/tensorflow_quantization/setup.py /workspace/TensorRT/tools/tensorflow-quantization
WORKDIR /workspace/TensorRT/tools/tensorflow-quantization
RUN python -m pip install .

RUN cd /opt
RUN mkdir trt_oss_src && \
   cd trt_oss_src && \
   echo "$PWD Building TRT OSS..." && \
   git clone -b release/8.4 https://github.com/nvidia/TensorRT TensorRT && \
   cd TensorRT && \
   git submodule update --init --recursive && \
   mkdir -p build && cd build  && \
   cmake .. -DGPU_ARCHS="35 52 53 60 61 70 75 80 86" -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out -DCUDA_VERSION=11.6 -DCUDNN_VERSION=8.4 && \
   make -j8 && \
   cp libnvinfer_plugin.so.8.4.* /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.8.4.1.5 && \
   cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/libnvinfer_plugin_static.a && \
   cp libnvonnxparser.so.8.4.* /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.8.4.1.5 && \
   cp libnvcaffeparser.so.8.4.* /usr/lib/x86_64-linux-gnu/libnvcaffeparser.so.8.4.1.5 && \
   cp trtexec /usr/local/bin/ && \
   cd ../../../ && \
   rm -rf trt_oss_src

RUN pip install https://urm.nvidia.com/artifactory/sw-eff-pypi-local/nvidia-eff/0.5.5/nvidia_eff-0.5.5-cp38-cp38-linux_x86_64.whl

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/