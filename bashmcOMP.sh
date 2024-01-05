#!/bin/bash

rm -f Results/montecarlo*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot montecarlo.cpp -o montecarlo.out
else
    g++ -fopenmp -lm montecarlo.cpp -o montecarlo.out
fi

if [ $? -eq 0 ]; then
    nprocess=(2 3 4 5 6)
    for process in "${nprocess[@]}"; do
        i=1024
        while [ $i -le 1000000 ]; do
            ./montecarlo.out $i $process
            ((i *= 2))
        done
    done
    python graph.py "montecarlo2D"
fi
