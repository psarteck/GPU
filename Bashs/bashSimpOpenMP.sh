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
    for ((process=1; process<=$nprocess; process++)); do
        echo "Calcul sur $process processeur(s)"
        i=10000
        while [ $i -le 100000000 ]; do
            $exe_folder/./simpsonOpenMP.out $i $process
            i=$((i * 10)) 
        done 
    done
    python3 $python_folder/graph.py "simp_Op_MP"
fi

