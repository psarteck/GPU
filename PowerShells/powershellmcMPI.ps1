$src_folder = "..\MPI"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content "..\parametres_Machine" | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item "$results_folder\montecarloMPI*" -Force

if ($env:OSTYPE -like "darwin*") {
    mpic++ -Xlinker -debug_snapshot "$src_folder\montecarloMPI.cpp" -o "$exe_folder\montecarloMPI.exe"
} else {
    mpic++ "$src_folder\montecarloMPI.cpp" -o "$exe_folder\montecarloMPI.exe"
}

if ($?) {
    for ($process = 2; $process -le $nprocess; $process*=2) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 2
        while ($i -le ([math]::Pow(2, 25) + 1)) {
            mpiexec -np $process "$exe_folder\montecarloMPI.exe" $i
            $i *= 2
        }
    }
    python3 "$python_folder\graph.py" "montecarlo_MPI"
}
