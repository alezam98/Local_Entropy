#!/usr/bin/env bash

set -e
source "/Data/alessandroz/myenv/bin/activate"

# Start the simulations
#for inputs_dir_raw in ...
#do
#	sed -i "2s/.*/$inputs_dir_raw/" inputs.txt
#	sleep 10 s
#	nohup python -u localentropy.py >& output_$inputs_dir_raw.out &
#done
nohup python -u localentropy.py >& output.out &