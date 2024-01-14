$src_folder = "..\MPI"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content "..\parametres_Machine" | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item "$results_folder\simp_MPI*" -Force

mpic++ "$src_folder\simpsonMPI.cpp" -o "$exe_folder\simpsonMPI.exe"

if ($?) {
    for ($process = 2; $process -le $nprocess; $process*=2) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 2
        while ($i -le ([math]::Pow(2, 32) + 1)) {
            mpiexec -np $process "$exe_folder\simpsonMPI.exe" $i $process
            $i *= 2
        }
    }
    python3 "$python_folder\graph.py" "simp_MPI"
}

