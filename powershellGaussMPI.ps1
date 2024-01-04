Remove-Item -Path Results\*.txt -Force

if ($env:OSTYPE -like "darwin*") {
    mpic++ -Xlinker -debug_snapshot gauss2DMPI.cpp -o gauss2DMPI
} else {
    mpic++ gauss2DMPI.cpp -o gauss2DMPI
}

if ($LASTEXITCODE -eq 0) {
    $nprocess = "2", "3", "4", "5", "6"
    foreach ($process in $nprocess) {
        $i = 1024
        while ($i -le 8400) {
            mpiexec -np $process gauss2DMPI.exe $i
            $i *= 2
        }
    }
    python3 graph.py "gauss2DMPI"
}
