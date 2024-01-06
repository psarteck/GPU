#!/bin/bash

src_folder="../MPI"
results_folder="../Results"
exe_folder="../Executables"
python_folder="../CodesPython"

rm $results_folder/montecarloMPI*

if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -Xlinker -debug_snapshot $src_folder/montecarloMPI.cpp -o $exe_folder/montecarloMPI.out
else
    mpic++ $src_folder/montecarloMPI.cpp -o $exe_folder/montecarloMPI.out
fi

if [ $? -eq 0 ]; then
    nprocess=("2" "3" "4" "5" "6")
    for process in "${nprocess[@]}"; do
        i=1024
        while [ $i -le 10000000 ]; do
            mpirun -np $process $exe_folder/./montecarloMPI.out $i
            ((i *= 2))
        done
    done
    python3 $python_folder/graph.py "montecarloMPI"
fi
