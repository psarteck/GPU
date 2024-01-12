# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc monte_carlo_cuda.cu -o monte_carlo_cuda.exe
# Exécution du code C++
./monte_carlo_cuda.exe
Remove-Item -Path "monte_carlo_cuda.lib", "monte_carlo_cuda.exe", "monte_carlo_cuda.exp" -Force
g++ monte_carlo.cpp
./a.exe
Remove-Item -Path "a.exe" -Force
# Exécution du script Python
python graph_cuda.py
Rename-Item -Path "error.png" -NewName "error_monte_carlo_cuda.png" -Force
Rename-Item -Path "time.png" -NewName "time_monte_carlo_cuda.png" -Force
Move-Item -Path "error_monte_carlo_cuda.png", "time_monte_carlo_cuda.png" -Destination "..\Images\" -Force
Remove-Item -Path "*.txt" -Force