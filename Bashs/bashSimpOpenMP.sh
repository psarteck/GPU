#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/simp_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp $src_folder/simpsonOpenMP.cpp -o $exe_folder/simpsonOpenMP.out
else
    g++ -fopenmp -lm $src_folder/simpsonOpenMP.cpp -o $exe_folder/simpsonOpenMP.out
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=10000
        while [ $i -le 100000000 ]; do
            $exe_folder/./simpsonOpenMP.out $i $process
            i=$((i * 10)) 
        done 
    done
    python3 $python_folder/graph.py "simp_Op_MP"
fi

