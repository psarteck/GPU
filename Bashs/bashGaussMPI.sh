#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/gauss_MPI*

if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -I /opt/homebrew/include/eigen3 -Xlinker -debug_snapshot $src_folder/gauss2DMPI.cpp -o $exe_folder/gauss2DMPI.out
else
    echo "Pas de solution de compilation sur Windows / Linux avec Eigen"
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 8400 ]; do
            mpirun -np $process $exe_folder/gauss2DMPI.out $i 
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "gauss_MPI"
fi

