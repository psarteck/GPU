#!/bin/bash

src_folder="../OpenMP"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

nprocess=$(awk -F'=' '/Nb_Procs/{print $2}' ../parametres_Machine | tr -d ' ')

rm $results_folder/montecarlo_Op_MP*

if [[ "$OSTYPE" == "darwin"* ]]; then
    omp -lstdc++ -Xlinker -debug_snapshot $src_folder/montecarloOpenMP.cpp -o $exe_folder/montecarloOp_MP.out
else
    g++ -fopenmp -lm $src_folder/montecarloOpenMP.cpp -o $exe_folder/montecarloOp_MP.out
fi

if [ $? -eq 0 ]; then
    for ((process=2; process<=$nprocess; process*=2)); do
        echo "Calcul sur $process processeur(s)"
        i=2
        while [ $i -le $((2**25 + 1)) ]; do
            $exe_folder/./montecarloOp_MP.out $i $process
            ((i *= 2))
        done
    done
    python3 $python_folder/graph.py "montecarlo_Op_MP"
fi
