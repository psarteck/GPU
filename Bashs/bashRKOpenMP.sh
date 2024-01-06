#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/RK_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot $src_folder/rungeKuttaOpenMP.cpp -o $exe_folder/rungeKuttaOpenMP.out
else
    g++ -fopenmp -lm $src_folder/rungeKuttaOpenMP.cpp -o $exe_folder/rungeKuttaOpenMP.out
fi

if [ $? -eq 0 ]; then
    for ((process=1; process<=$nprocess; process++)); do
        echo "Calcul sur $process processeur(s)"
        i=1024
        while [ $i -le 35000 ]; do
            $exe_folder/./rungeKuttaOpenMP.out $i $process
            i=$((i * 2)) 
        done 
    done
fi
python3 $python_folder/graph.py "RK_Op_MP" 

