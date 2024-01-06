#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/gauss_MPI*

if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -I /opt/homebrew/include/eigen3 -Xlinker -debug_snapshot $src_folder/gauss2DMPI.cpp -o $exe_folder/gauss2DMPI.out
else
    echo "Pas de solution de compilation sur Windows / Linux avec Eigen"
fi

if [ $? -eq 0 ]; then
    for ((process=1; process<=$nprocess; process++)); do
        i=1024
        echo "Calcul sur $process processeur(s)"
        while [ $i -le 8400 ]; do
            mpirun -np $process $exe_folder/gauss2DMPI.out $i 
            i=$((i * 2)) 
        done 
    done
    python3 $python_folder/graph.py "gauss_MPI"
fi


