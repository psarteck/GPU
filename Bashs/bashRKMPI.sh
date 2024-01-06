#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/RK_MPI*

mpic++ -std=c++11 $src_folder/rungeKuttaMPI.cpp -o $exe_folder/rungeKuttaMPI.out

if [ $? -eq 0 ]; then
    for ((process=1; process<=$nprocess; process++)); do
        echo "Calcul sur $process processeur(s)"
        i=1024
        while [ $i -le 35000 ]; do
            mpirun -np $process $exe_folder/./rungeKuttaMPI.out $i
            i=$((i * 2)) 
        done 
    done
fi
python3 $python_folder/graph.py "RK_MPI" 



