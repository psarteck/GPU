Remove-Item -Path Results\error_mesh* -Force

if ($env:OSTYPE -like "darwin*") {
    omp -lstdc++ -Xlinker -debug_snapshot rungeKuttaOpenMP.cpp -o rungeKuttaOpenMP.exe
} else {
    g++ -fopenmp -lm rungeKuttaOpenMP.cpp -o rungeKuttaOpenMP.exe
}

if ($LASTEXITCODE -eq 0) {
    $nprocess = @(2, 3, 4, 5, 6)
    foreach ($process in $nprocess) {
        $i = 1024
        while ($i -le 250000) {
            .\rungeKuttaOpenMP.exe $i $process
            $i *= 2
        }
    }
}

python .\graph.py "error_mesh"
