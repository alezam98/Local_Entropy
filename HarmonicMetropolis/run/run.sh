#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv_3.9.14_11.7/bin/activate"

protname=`sed -n '1p' < inputs/inputs.txt`
T=`sed -n '9p' < inputs/inputs.txt`
dx=`sed -n '10p' < inputs/inputs.txt`
s=`sed -n '11p' < inputs/inputs.txt`
file_id=$protname-T$T-dx$dx-s$s

nohup python -u metropolis.py >& outputs/$file_id.out &
