Write-Host "Hello, PowerShell!"
# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc simpson_cuda.cu -o simpson_cuda.exe

# Exécution du code C++
./simpson_cuda.exe

# Exécution du script Python
python graph_cuda.py
