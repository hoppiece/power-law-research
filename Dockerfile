FROM  nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /code

RUN apt-get update && \
  apt-get --yes --no-install-recommends install \
  git \
  htop 


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt