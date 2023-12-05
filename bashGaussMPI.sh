#!/bin/sh

rm Results/*.txt
if [[ "$OSTYPE" == "darwin"* ]]; then
    mpic++ -I /opt/homebrew/include/eigen3 -Xlinker -debug_snapshot gauss2DMPI.cpp -o gauss2DMPI.out
else
    echo "Pas de solution de compilation sur Windows / Linux"
fi

if [ $? -eq 0 ]; then
    nprocess="2 3 4 5 6"
    for process in $nprocess; do
        i=1024
        while [ $i -le 8400 ]; do
            mpirun -np $process gauss2DMPI.out $i 
            # ./gauss2DMPI $i $process
            i=$((i * 2)) 
        done 
    done
    python3 graph.py "gauss2DMPI"
fi

