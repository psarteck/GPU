# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc RK4_cuda.cu -o RK4_cuda.exe
# Exécution du code C++
./RK4_cuda.exe
Remove-Item -Path "RK4_cuda.lib", "RK4_cuda.exe", "RK4_cuda.exp" -Force
g++ RK4.cpp
./a.exe
Remove-Item -Path "a.exe" -Force
# Exécution du script Python
python graph_cuda.py
Rename-Item -Path "error.png" -NewName "error_RK4_cuda.png" -Force
Rename-Item -Path "time.png" -NewName "time_RK4_cuda.png" -Force
Move-Item -Path "error_RK4_cuda.png", "time_RK4_cuda.png" -Destination "..\Images\" -Force
Remove-Item -Path "*.txt" -Force