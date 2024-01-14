# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc Gauss2D_cuda.cu -o Gauss2D_cuda.exe -w
# Exécution du code C++
./Gauss2D_cuda.exe
Remove-Item -Path "Gauss2D_cuda.lib", "Gauss2D_cuda.exe", "Gauss2D_cuda.exp" -Force
g++ -w Gauss2D.cpp
./a.exe
Remove-Item -Path "a.exe" -Force
# Exécution du script Python
python graph_cuda.py gauss
Move-Item -Path "output_gauss_cuda.txt" -Destination "..\Results\" -Force
Remove-Item -Path "*.txt" -Force