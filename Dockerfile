FROM  nvcr.io/nvidia/pytorch:23.10-py3
# FROM nvidia/cuda:12.1.1-base-ubuntu22.04


WORKDIR /code

# Set timezone
ENV TZ=Asia/Tokyo
ENV LANG C.UTF-8


RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends install \
  automake \
  build-essential \
  git \
  libcurl4-openssl-dev \
  libfuse-dev \
  libssl-dev  \
  libtool \
  libxml2-dev \
  mime-support \
  pkg-config \
  sudo \
  htop \
  vim \
  tmux \
  emacs \
  openssh-server \
  zip \
  wget \
  curl \
  bzip2 \
  unzip \
  pipx \
  parallel \
  &&  apt-get clean \
  && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --upgrade --no-cache-dir pip setuptools && \
  pip install -r requirements.txt
