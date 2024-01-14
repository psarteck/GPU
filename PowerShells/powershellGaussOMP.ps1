$src_folder = "..\OpenMP"
$results_folder = "..\Results"
$exe_folder = "..\Executables"
$python_folder = "..\CodesPython"

$nprocess = (Get-Content ..\parametres_Machine | Select-String -Pattern "Nb_Procs").ToString().Split('=')[1].Trim()

Remove-Item $results_folder\gauss_Op_MP* -Force

if ($env:OSTYPE -like "darwin*") {
    eig -fopenmp -lstdc++ -Xlinker -debug_snapshot $src_folder\gauss2DOpenMP.cpp -o $exe_folder\gauss2DOpenMP.exe
} else {
    g++ -fopenmp -lm $src_folder\gauss2DOpenMP.cpp -o $exe_folder\gauss2DOpenMP.exe
}

if ($?) {
    for ($process = 2; $process -le $nprocess; $process*=2) {
        Write-Output "Calcul sur $process processeur(s)"
        $i = 2
        while ($i -le ([math]::Pow(2, 14) + 1)) {
            & "$exe_folder\gauss2DOpenMP.exe" $i $process
            $i *= 2
        }
    }
    python3 "$python_folder\graph.py" "gauss_Op_MP"
}
