#!/bin/sh

rm *.txt
if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ jacobi.c++ -o simpsonMPI.out
else
    g++ -fopenmp -lm simpson.cpp -o simpson
fi

if [ $? -eq 0 ]; then
    nprocess="2 4 5 6"
    for process in $nprocess; do
        i=10000
        while [ $i -le 100000000 ]; do
            mpirun -np $process ./simpsonMPI.out $i $process
            i=$((i * 10)) 
        done 
    done
    python3 graph.py 
    # for process in $nprocess; do
        # python3 graph.py $process
    # done
fi

