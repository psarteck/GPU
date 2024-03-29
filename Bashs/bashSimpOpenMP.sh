#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/simp_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp $src_folder/simpsonOpenMP.cpp -o $exe_folder/simpsonOpenMP.out
else
    g++ -fopenmp -lm $src_folder/simpsonOpenMP.cpp -o $exe_folder/simpsonOpenMP.out
fi

if [ $? -eq 0 ]; then
    for ((process=2; process<=$nprocess; process*=2)); do
        echo "Calcul sur $process processeur(s)"
        i=2
        while [ $i -le $((2**30 + 1)) ]; do
            $exe_folder/./simpsonOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "simp_Op_MP"
fi

