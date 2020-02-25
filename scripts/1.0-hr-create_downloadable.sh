#!/bin/bash

declare -a dirs=("train" "val" "test")
input_dir="."
single=false
output_dir="."

if single=true; then
	for dir in "${dirs[@]}"
	do
		cp -r $input_dir/$dir $output_dir
	done
fi

if single=false; then
	for dir in  "${dirs[@]}"
    do
        tar -czvf $output_dir/$dir.tar.gz $input_dir/$dir/* 
    done
fi
