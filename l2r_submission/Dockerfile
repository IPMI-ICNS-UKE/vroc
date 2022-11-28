FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3


WORKDIR /l2r

COPY l2r_submission/requirements.txt .
RUN pip3 install -r requirements.txt

RUN echo

COPY l2r_submission/inference.py .

COPY l2r_submission/test.sh .
COPY l2r_submission/train.sh .

COPY vroc/ ./vroc
COPY setup.py .

RUN pip3 install .

RUN chmod +x test.sh
RUN chmod +x train.sh
