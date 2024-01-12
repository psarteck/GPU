# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc simpson_cuda.cu -o simpson_cuda.exe
# Exécution du code C++
./simpson_cuda.exe
Remove-Item -Path "simpson_cuda.lib", "simpson_cuda.exe", "simpson_cuda.exp" -Force
g++ simpson.cpp
./a.exe
Remove-Item -Path "a.exe" -Force
# Exécution du script Python
python graph_cuda.py
Rename-Item -Path "error.png" -NewName "error_simpson_cuda.png" -Force
Rename-Item -Path "time.png" -NewName "time_simpson_cuda.png" -Force
Move-Item -Path "error_simpson_cuda.png", "time_simpson_cuda.png" -Destination "..\Images\" -Force
Remove-Item -Path "*.txt" -Force