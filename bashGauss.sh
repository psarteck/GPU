#!/bin/sh

rm Results/*.txt
if [[ "$OSTYPE" == "darwin"* ]]; then
    eig -fopenmp -lstdc++ -Xlinker -debug_snapshot gauss2D.cpp -o gauss2D.out
else
    g++ -fopenmp -lm simpson.cpp -o simpson 
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=256
        while [ $i -le 4200 ]; do
            ./gauss2D.out $i $process
            i=$((i * 2)) 
        done 
    done
    python3 graph.py "gauss2D"
    # for process in $nprocess; do
        # python3 graph.py $process
    # done
fi

