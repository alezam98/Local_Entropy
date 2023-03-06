#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv/bin/activate"

protname=`sed -n '1p' < inputs/inputs.txt`
T=`sed -n '7p' < inputs/inputs.txt`
d=`sed -n '8p' < inputs/inputs.txt`
file_id=$protname-T$T-d$d

nohup python -u wall_metropolis.py >& outputs/$file_id.out &
