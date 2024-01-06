# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc Gauss2D_cuda.cu -o Gauss2D_cuda.exe
# Exécution du code C++
./Gauss2D_cuda.exe
Remove-Item -Path "Gauss2D_cuda.lib", "Gauss2D_cuda.exe", "Gauss2D_cuda.exp" -Force
# Exécution du script Python
python graph_cuda.py
Rename-Item -Path "error.png" -NewName "error_Gauss2D_cuda.png" -Force
Rename-Item -Path "time.png" -NewName "time_Gauss2D_cuda.png" -Force
Move-Item -Path "error_Gauss2D_cuda.png", "time_Gauss2D_cuda.png" -Destination "..\Images\" -Force
Remove-Item -Path "error_results.txt", "time_results.txt" -Force