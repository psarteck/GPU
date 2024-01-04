#!/bin/sh

rm Results/error_mesh*
if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot rungeKuttaOpenMP.cpp -o rungeKuttaOpenMP.out
else
    g++ -fopenmp -lm rungeKuttaOpenMP.cpp -o rungeKuttaOpenMP.out
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 250000 ]; do
            ./rungeKuttaOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
fi
python3 graph.py "error_mesh" 

