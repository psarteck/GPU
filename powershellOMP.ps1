Remove-Item *.txt -ErrorAction SilentlyContinue

if ($env:OSTYPE -like "darwin*") {
    omp simpson.cpp -o simpson
} else {
    g++ -fopenmp -lm simpson.cpp -o simpson
}

if ($LastExitCode -eq 0) {
    $nprocess = @(2, 3, 4, 5, 6)
    foreach ($process in $nprocess) {
        $i = 10000
        while ($i -le 100000000) {
            .\simpson $i $process
            $i = $i * 10
        }
    }
    python graph.py
    # foreach ($process in $nprocess) {
    #     python graph.py $process
    # }
}
