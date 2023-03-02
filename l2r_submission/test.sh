#!/bin/bash
JSON_PATH=$1
GPU=$2

python inference.py ${JSON_PATH} ${GPU}
