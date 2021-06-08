#!/bin/zsh
filename=$1
save_path=$2
for line in $(cat $filename); do
	scp gideon@192.168.100.53:$line $save_path ;
done
