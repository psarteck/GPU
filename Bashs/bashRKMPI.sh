#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/RK_MPI*

mpic++ -std=c++11 $src_folder/rungeKuttaMPI.cpp -o $exe_folder/rungeKuttaMPI.out

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 35000 ]; do
            mpirun -np $process $exe_folder/./rungeKuttaMPI.out $i
            i=$((i * 2)) 
        done 
    done
fi
python3 $python_folder/graph.py "RK_MPI" 



