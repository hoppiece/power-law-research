FROM  nvcr.io/nvidia/pytorch:23.04-py3
# FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# https://github.com/hadolint/hadolint/releases
ARG HADOLINT_VERSION=2.12.0

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
  parallel \
  &&  apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install hadolint
RUN curl -L https://github.com/hadolint/hadolint/releases/download/v$HADOLINT_VERSION/hadolint-`uname -s`-`uname -m` -o /usr/local/bin/hadolint && \
  chmod +x /usr/local/bin/hadolint

RUN pip install --upgrade --no-cache-dir pip setuptools wheel && \
  pip install --no-cache-dir poetry && \
  poetry config virtualenvs.in-project false


COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-interaction --no-ansi -vvv

