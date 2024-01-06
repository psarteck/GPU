#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/montecarlo_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot $src_folder/montecarloOpenMP.cpp -o $exe_folder/montecarloOp_MP.out
else
    g++ -fopenmp -lm $src_folder/montecarloOpenMP.cpp -o $exe_folder/montecarloOp_MP.out
fi

if [ $? -eq 0 ]; then
    nprocess=(2 3 4 5 6)
    for process in "${nprocess[@]}"; do
        i=1024
        while [ $i -le 1000000 ]; do
            $exe_folder/./montecarloOp_MP.out $i $process
            ((i *= 2))
        done
    done
    python3 $python_folder/graph.py "montecarlo_Op_MP"
fi
