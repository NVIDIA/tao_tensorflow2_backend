ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:21.09-tf2-py3
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

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"
ENV PYTHONPATH='/workspace/tao-tf2'

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
