$src_folder = "..\MPI"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content "..\parametres_Machine" | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item "$results_folder\RK_MPI*" -Force

mpic++ "$src_folder\rungeKuttaMPI.cpp" -o "$exe_folder\rungeKuttaMPI.exe"

if ($?) {
    for ($process = 1; $process -le $nprocess; $process++) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 1024
        while ($i -le 35000) {
            mpiexec -np $process "$exe_folder\rungeKuttaMPI.exe" $i
            $i *= 2
        }
    }
}
python3 "$python_folder\graph.py" "RK_MPI"
