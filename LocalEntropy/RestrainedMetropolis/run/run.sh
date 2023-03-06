#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv/bin/activate"

simtype=`sed -n '1p' < inputs/inputs.txt`
protname=`sed -n '2p' < inputs/inputs.txt`
T=`sed -n '8p' < inputs/inputs.txt`
g=`sed -n '9p' < inputs/inputs.txt`

if [ $simtype == 'SM' ]
then
	file_id=$simtype-$protname-T$T-g$g
elif [ $simtype == 'MM' ]
then
	file_id=$simtype-$protname-T$T
else
	echo 'Wrong simtype value. Exit.'
	exit 1
fi

nohup python -u metropolis.py >& outputs/$file_id.out &
