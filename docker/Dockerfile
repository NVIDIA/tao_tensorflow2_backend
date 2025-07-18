ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:25.01-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# upgrade pip
RUN pip install --upgrade pip

# Copy requirement and install
WORKDIR /workspace/
COPY docker/requirements.txt /workspace/
COPY docker/requirements-apt.txt /workspace/
RUN pip install -r requirements.txt && \
    rm -rf requirements.txt

# Deprecated since DLFW 25.01
# Install public version of Nvidia QAT Toolkit. Due to issues with the public version, reverting the problematic commit
RUN git clone https://github.com/NVIDIA/TensorRT.git
COPY docker/tensorflow_quantization/setup.py /workspace/TensorRT/tools/tensorflow-quantization
WORKDIR /workspace/TensorRT/tools/tensorflow-quantization
RUN git revert --no-commit c46089ff8b63578dc2edfb993e7043cb4fb7cde6
RUN python -m pip install .

# TAO-CORE microservice dependencies
WORKDIR /workspace/k8s
RUN wget https://github.com/kubernetes-client/python/archive/refs/tags/v23.6.0.zip && \
    unzip v23.6.0.zip && \
    cd python-23.6.0 && \
    sed -i "s/PACKAGE_NAME\ \=\ .*/PACKAGE_NAME = \"kubernetes-client\"/" setup.py && \
    pip install --disable-pip-version-check -r requirements.txt && \
    python setup.py install
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes $(cat /workspace/requirements-apt.txt)
RUN rm -rf /workspace/requirements-apt.txt

ENV TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false tf_xla_async_io_level=0"

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/