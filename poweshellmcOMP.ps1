Remove-Item -Path Results\montecarlo* -Force

if ($env:OSTYPE -like "darwin*") {
    omp -lstdc++ -Xlinker -debug_snapshot montecarlo.cpp -o montecarlo.exe
} else {
    g++ -fopenmp -lm montecarlo.cpp -o montecarlo.exe
}

if ($LASTEXITCODE -eq 0) {
    $nprocess = @(2, 3, 4, 5, 6)
    foreach ($process in $nprocess) {
        $i = 1024
        while ($i -le 1000000) {
            .\montecarlo.exe $i $process
            $i *= 2
        }
    }
    python .\graph.py "montecarlo2D"
}

