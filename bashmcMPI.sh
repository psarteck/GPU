#!/bin/bash

rm -f Results/*.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -Xlinker -debug_snapshot montecarloMPI.cpp -o montecarloMPI
else
    mpic++ montecarloMPI.cpp -o montecarloMPI
fi

if [ $? -eq 0 ]; then
    nprocess=("2" "3" "4" "5" "6")
    for process in "${nprocess[@]}"; do
        i=1024
        while [ $i -le 10000000 ]; do
            mpirun -np $process montecarloMPI.out $i
            ((i *= 2))
        done
    done
    python3 graph.py "montecarloMPI"
fi
