#!/usr/bin/env bash

set -e
source "$HOME/workspace/myenv_3.9.14_11.7/bin/activate"

protname=`sed -n '1p' < inputs/inputs.txt`
y=`sed -n '8p' < inputs/inputs.txt`
s=`sed -n '9p' < inputs/inputs.txt`
T=`sed -n '4p' < inputs/inputs.txt`

l=`sed -n '15p' < inputs/inputs.txt`
e=`sed -n '10p' < inputs/inputs.txt`
if [ $l == '1' ]
then
	c=`sed -n '16p' < inputs/inputs.txt`
	file_id=$protname-y$y-s$s-c$c-T$T
else
	file_id=$protname-y$y-s$s-e$e-T$T
fi

ng=`sed -n '7p' < inputs/inputs.txt`
if [ $ng == '1' ]
then
	g=`sed -n '5p' < inputs/inputs.txt`
        file_id=$file_id-g$g
else
        file_id=$file_id
fi

nohup python -u metropolis.py >& outputs/$file_id.out &
