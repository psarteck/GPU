#!/bin/sh

rm Results/error_mesh_MPI*

mpic++ -std=c++11 rungeKuttaMPI.cpp -o rungeKuttaMPI.out


if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 150000 ]; do
            mpirun -np $process ./rungeKuttaMPI.out $i
            i=$((i * 2)) 
        done 
    done
fi
python3 graph.py "error_mesh_MPI" 



