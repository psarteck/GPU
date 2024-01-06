#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/gauss_Op_MP*
if [[ "$OSTYPE" == "darwin"* ]]; then
    eig -fopenmp -lstdc++ -Xlinker -debug_snapshot $src_folder/gauss2DOpenMP.cpp -o $exe_folder/gauss2DOpenMP.out
else
    g++ -fopenmp -lm $src_folder/gauss2DOpenMP.cpp -o $exe_folder/gauss2DOpenMP.out
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=256
        while [ $i -le 4200 ]; do
            $exe_folder/./gauss2DOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "gauss_Op_MP"
fi

