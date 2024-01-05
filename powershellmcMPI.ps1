Remove-Item -Path Results\*.txt -Force

if ($env:OSTYPE -like "darwin*") {
    mpic++ -Xlinker -debug_snapshot montecarloMPI.cpp -o montecarloMPI
} else {
    mpic++ montecarloMPI.cpp -o montecarloMPI
}

if ($LASTEXITCODE -eq 0) {
    $nprocess = "2", "3", "4", "5", "6"
    foreach ($process in $nprocess) {
        $i = 1024
        while ($i -le 10000000) {
            mpiexec -np $process montecarloMPI.exe $i
            $i *= 2
        }
    }
    python3 graph.py "montecarloMPI"
}