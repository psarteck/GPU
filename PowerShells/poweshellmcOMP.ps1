$src_folder = "..\OpenMP"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content "..\parametres_Machine" | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item "$results_folder\montecarlo_Op_MP*" -Force

if ($env:OSTYPE -like "darwin*") {
    omp -lstdc++ -Xlinker -debug_snapshot "$src_folder\montecarloOpenMP.cpp" -o "$exe_folder\montecarloOp_MP.exe"
} else {
    g++ -fopenmp -lm "$src_folder\montecarloOpenMP.cpp" -o "$exe_folder\montecarloOp_MP.exe"
}

if ($?) {
    for ($process = 2; $process -le $nprocess; $process*=2) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 2
        while ($i -le ([math]::Pow(2, 25) + 1)) {
            & "$exe_folder\montecarloOp_MP.exe" $i $process
            $i *= 2
        }
    }
    python3 "$python_folder\graph.py" "montecarlo_Op_MP"
}
