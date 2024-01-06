$src_folder = "..\MPI"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content ..\parametres_Machine | Select-String -Pattern "Nb_Procs").Split('=')[1].Trim()

Remove-Item $results_folder\gauss_MPI* -Force

if ($env:OSTYPE -like "darwin*") {
    mpic++ -I /opt/homebrew/include/eigen3 -Xlinker -debug_snapshot $src_folder\gauss2DMPI.cpp -o $exe_folder\gauss2DMPI.exe
} else {
    Write-Output "Pas de solution de compilation sur Windows / Linux avec Eigen"
}

if ($?) {
    for ($process = 1; $process -le $nprocess; $process++) {
        $i = 1024
        Write-Output "Calcul sur $process processeur(s)"
        while ($i -le 8400) {
            mpiexec -np $process $exe_folder\gauss2DMPI.exe $i
            $i *= 2
        }
    }
    python3 $python_folder\graph.py "gauss_MPI"
}



