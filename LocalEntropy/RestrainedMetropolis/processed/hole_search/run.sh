#!/usr/bin/env bash

set -e
source "$HOME/Desktop/myenv/bin/activate"

dtype=`sed -n '3p' < inputs/main_inputs.txt`
T=`sed -n '1p' < inputs/main_inputs.txt`
e=`sed -n '2p' < inputs/main_inputs.txt`
p=`sed -n '5p' < inputs/main_inputs.txt`
s=`sed -n '4p' < inputs/main_inputs.txt`
dm=`sed -n '5p' < inputs/main_inputs.txt`
file_id=$dtype-T$T-e$e-p$p-s$s-dm$dm

nohup python -u main.py >& outputs/$file_id.out &
