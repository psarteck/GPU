#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/montecarloMPI*

if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -Xlinker -debug_snapshot $src_folder/montecarloMPI.cpp -o $exe_folder/montecarloMPI.out
else
    mpic++ $src_folder/montecarloMPI.cpp -o $exe_folder/montecarloMPI.out
fi

if [ $? -eq 0 ]; then
    for ((process=1; process<=$nprocess; process++)); do
        echo "Calcul sur $process processeur(s)"
        i=1024
        while [ $i -le 10000000 ]; do
            mpirun -np $process $exe_folder/./montecarloMPI.out $i
            ((i *= 2))
        done
    done
    python3 $python_folder/graph.py "montecarlo_MPI"
fi
