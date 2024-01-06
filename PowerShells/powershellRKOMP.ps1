$src_folder = "..\OpenMP"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content "..\parametres_Machine" | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item "$results_folder\RK_Op_MP*" -Force

if ($env:OSTYPE -like "darwin*") {
    omp -lstdc++ -Xlinker -debug_snapshot "$src_folder\rungeKuttaOpenMP.cpp" -o "$exe_folder\rungeKuttaOpenMP.exe"
} else {
    g++ -fopenmp -lm "$src_folder\rungeKuttaOpenMP.cpp" -o "$exe_folder\rungeKuttaOpenMP.exe"
}

if ($?) {
    for ($process = 1; $process -le $nprocess; $process++) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 1024
        while ($i -le 35000) {
            & "$exe_folder\rungeKuttaOpenMP.exe" $i $process
            $i *= 2
        }
    }
}
python3 "$python_folder\graph.py" "RK_Op_MP"

