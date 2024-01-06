#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/RK_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot $src_folder/rungeKuttaOpenMP.cpp -o $exe_folder/rungeKuttaOpenMP.out
else
    g++ -fopenmp -lm $src_folder/rungeKuttaOpenMP.cpp -o $exe_folder/rungeKuttaOpenMP.out
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 35000 ]; do
            $exe_folder/./rungeKuttaOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
fi
python3 $python_folder/graph.py "RK_Op_MP" 

