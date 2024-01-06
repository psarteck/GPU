#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/simp_MPI*

mpic++ -std=c++11 $src_folder/simpsonMPI.cpp -o $exe_folder/simpsonMPI.out

if [ $? -eq 0 ]; then
    for ((process=1; process<=$nprocess; process++)); do
        echo "Calcul sur $process processeur(s)"
        i=10000
        while [ $i -le 100000000 ]; do
            mpirun -np $process $exe_folder/./simpsonMPI.out $i $process
            i=$((i * 10)) 
        done 
    done
    python3 $python_folder/graph.py "simp_MPI"
fi

