# Compilation du code C++ avec nvcc (compiler CUDA)
nvcc simpson_cuda.cu -o simpson_cuda.exe
# Exécution du code C++
./simpson_cuda.exe
Remove-Item -Path "simpson_cuda.lib", "simpson_cuda.exe", "simpson_cuda.exp" -Force
g++ ../Sequentiel/simpson.cpp
./a.exe
Remove-Item -Path "a.exe" -Force
# Exécution du script Python
python graph_cuda.py simp
python graph_all.py simp