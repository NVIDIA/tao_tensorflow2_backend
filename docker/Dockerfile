ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:21.09-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy detectron code and build
WORKDIR /workspace/
COPY . .
RUN pip install -r requirements.txt

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"
RUN pip install git+https://github.com/NVIDIA/dllogger
RUN pip install tensorflow-addons==0.13.0
ENV PYTHONPATH='/workspace'