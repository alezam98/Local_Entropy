#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv/bin/activate"

simtype=`sed -n '1p' < inputs/inputs.txt`
protname=`sed -n '2p' < inputs/inputs.txt`
T=`sed -n '8p' < inputs/inputs.txt`
g=`sed -n '9p' < inputs/inputs.txt`
file_id=$simtype-$protname-T$T-g$g
# add if statement for multimetropolis

nohup python -u metropolis.py >& outputs/$file_id.out &
