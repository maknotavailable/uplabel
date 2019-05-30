#!/bin/bash

# Activate Environment
source /home/nbcommon/anaconda3_510/bin/activate

# Set up proxy
http_proxy=http://webproxy:3128
https_proxy=http://webproxy:31128
export http_proxy
export https_proxy

# pip
pip install openpyxl
pip install pyyaml
pip install xlrd

# conda
config --add channels conda-forge
conda install spacy=2.1.3 -y

# spacy model download 
python -m spacy download de
