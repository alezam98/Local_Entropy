#!/usr/bin/env bash

set -e
source "/Data/alessandroz/myenv/bin/activate"

# Start the simulations
#for gamma in 4 5 6
#do
#	sed -i "5s/.*/$gamma./" inputs.txt
#	sed -i "6s/.*/$gamma./" inputs.txt
#	sleep 10 s
#	nohup python -u production.py >& outputs/output_$gamma.out &
#done
nohup python -u production.py >& outputs/output4.out &
