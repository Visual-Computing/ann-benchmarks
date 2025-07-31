#!/bin/bash

algo="deglib"
if [ "$2" == "hnsw" ]; then
	algo="hnswlib"
fi

case "$1" in
	i)
		python install.py --algorithm "$algo"
		;;
	r)
		shift
		python run.py --dataset fashion-mnist-784-euclidean --algorithm "$algo" --runs 1
		;;
	p)
		sudo chown -R "$USER:$USER" results/

		python plot.py --x-scale logit --y-scale log --dataset fashion-mnist-784-euclidean
		silent gwenview results/fashion-mnist-784-euclidean.png
		;;
	*)
		echo "invalid option"
		;;
esac
