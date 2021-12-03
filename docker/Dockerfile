FROM nvidia/cuda:10.1-devel-ubuntu18.04

COPY ./davarocr /app/davarocr
COPY ./models /app/models
COPY ./src /app/src
COPY ./requirements.in /app/
COPY ./setup_docker.sh /app/
COPY ./readme.md /app/

WORKDIR /app

RUN apt-get update && apt-get -y upgrade
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt install -y gcc \
    g++ \
    python3 \
    python3-pip \
    python3-dev \
    python3-opencv \
    libopencv-dev

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN ./setup_docker.sh

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["uvicorn", "src.web:app", "--host", "0.0.0.0", "--port", "8000"]
