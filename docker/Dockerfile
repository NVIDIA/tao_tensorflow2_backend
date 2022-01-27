ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com:5005/dl/dgx/tensorflow:master-py3-devel
FROM ${FROM_IMAGE_NAME}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy detectron code and build
WORKDIR /workspace/
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt && \
    rm -rf requirements.txt

COPY tensorrt_qat /tmp/tensorrt_qat
WORKDIR /tmp/tensorrt_qat
RUN pip install .
COPY tensorflow-onnx /tmp/tensorflow-onnx
WORKDIR /tmp/tensorflow-onnx
RUN pip install .

RUN pip install https://urm.nvidia.com/artifactory/sw-eff-pypi-local/nvidia-eff/0.5.5/nvidia_eff-0.5.5-cp38-cp38-linux_x86_64.whl

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"
ENV PYTHONPATH='/workspace/tao-tf2'

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
