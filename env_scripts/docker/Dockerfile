# ARG UBUNTU_VERSION=18.04
ARG UBUNTU_VERSION=20.04

FROM ubuntu:${UBUNTU_VERSION} AS base

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        rsync \
        software-properties-common \
        sudo \
        unzip \
        zip \
        zlib1g-dev \
        openjdk-8-jdk \
        openjdk-8-jre-headless \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CI_BUILD_PYTHON python

# CACHE_STOP is used to rerun future commands, otherwise cloning tensorflow will be cached and will not pull the most recent version
ARG CACHE_STOP=1
# Check out TensorFlow source code if --build-arg CHECKOUT_TF_SRC=1
ARG CHECKOUT_TF_SRC=0
# In case of Python 2.7+ we need to add passwd entries for user and group id
RUN chmod a+w /etc/passwd /etc/group
RUN test "${CHECKOUT_TF_SRC}" -eq 1 && git clone https://github.com/tensorflow/tensorflow.git /tensorflow_src || true

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    openjdk-8-jdk \
    python3-dev \
    virtualenv \
    swig

RUN python3 -m pip --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    matplotlib \
    mock \
    'numpy==1.19.2' \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    tqdm \
    enum34

# cmake
ARG CMAKE_VERSION=3.21.0

RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

# Install bazel
# ARG BAZEL_VERSION=3.7.2
# RUN mkdir /bazel && \
#     wget -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
#     wget -O /bazel/LICENSE.txt "https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE" && \
#     chmod +x /bazel/installer.sh && \
#     /bazel/installer.sh && \
#     rm -f /bazel/installer.sh

COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc