#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/simp_MPI*

mpic++ -std=c++11 $src_folder/simpsonMPI.cpp -o $exe_folder/simpsonMPI.out

if [ $? -eq 0 ]; then
    for ((process=2; process<=$nprocess; process*=2)); do
        echo "Calcul sur $process processeur(s)"
        i=16
        while [ $i -le $((2**30 + 1)) ]; do
            mpirun -np $process $exe_folder/./simpsonMPI.out $i $process
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "simp_MPI"
fi

