#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/simp_MPI*

mpic++ -std=c++11 $src_folder/simpsonMPI.cpp -o $exe_folder/simpsonMPI.out

if [ $? -eq 0 ]; then
    nprocess="2 4 5 6"
    for process in $nprocess; do
        i=10000
        while [ $i -le 100000000 ]; do
            mpirun -np $process $exe_folder/./simpsonMPI.out $i $process
            i=$((i * 10)) 
        done 
    done
    python3 $python_folder/graph.py "simp_MPI"
fi

