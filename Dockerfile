FROM  nvcr.io/nvidia/pytorch:22.12-py3

USER root
WORKDIR /code

# Set timezone
ENV TZ=Asia/Tokyo
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
  apt-get --yes --no-install-recommends install \
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
  parallel \
  &&  apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install hadolint
RUN curl -L https://github.com/hadolint/hadolint/releases/download/v2.10.0/hadolint-Linux-x86_64 -o /usr/local/bin/hadolint && \
  chmod +x /usr/local/bin/hadolint


COPY requirements.txt requirements.txt
RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
  pip install --no-cache-dir -r requirements.txt && \
  rm requirements.txt
