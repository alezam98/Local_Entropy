#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv/bin/activate"

# Start the simulations
#for protein_name in ...
#do
#	sed -i "1s/.*/$protein_name/" inputs.txt
#	sleep 10 s
#	nohup python -u localentropy.py >& output_$protein_name.out &
#done
nohup python -u localentropy.py >& output.out &
