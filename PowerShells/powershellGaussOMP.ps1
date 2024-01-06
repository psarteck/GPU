Remove-Item -Path Results\*.txt -Force

if ($env:OSTYPE -like "darwin*") {
    eig -fopenmp -lstdc++ -Xlinker -debug_snapshot gauss2D.cpp -o gauss2D
} else {
    g++ -fopenmp -lm gauss2D.cpp -o gauss2D
}

if ($LASTEXITCODE -eq 0) {
    $nprocess = @(2, 3, 4, 5, 6)
    foreach ($process in $nprocess) {
        $i = 256
        while ($i -le 4200) {
            .\gauss2D.exe $i $process
            $i *= 2
        }
    }
    python3 graph.py "gauss2D"
    # for ($i = 0; $i -lt $nprocess.Length; $i++) {
    #     python3 graph.py $nprocess[$i]
    # }
}
