#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/gauss_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    eig -fopenmp -lstdc++ -Xlinker -debug_snapshot $src_folder/gauss2DOpenMP.cpp -o $exe_folder/gauss2DOpenMP.out
else
    g++ -fopenmp -lm $src_folder/gauss2DOpenMP.cpp -o $exe_folder/gauss2DOpenMP.out
fi

if [ $? -eq 0 ]; then
    for ((process=2; process<=$nprocess; process*=2)); do
        echo "Calcul sur $process processeur(s)"
        i=16
        while [ $i -le $((2**14 + 1)) ]; do
            $exe_folder/./gauss2DOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "gauss_Op_MP"
fi

