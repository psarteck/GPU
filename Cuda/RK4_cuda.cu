#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>

const double PI = 3.14159265358979323846;  // Utiliser une valeur explicite pour PI

// Fonction représentant la solution exacte de l'équation de la chaleur
double exactSolution(double alpha, double x, double t) {
    return 1.0 / sqrt(4.0 * PI * alpha * t) * exp(-x * x / (4.0 * alpha * t));
}

// Fonction représentant l'équation de la chaleur
__device__ double heatEquation(double alpha, const double* u, int i) {
    return alpha * (u[i - 1] - 2 * u[i] + u[i + 1]);
}

// Kernel CUDA pour la résolution de l'équation de la chaleur avec RK4
__global__ void rungeKuttaHeatEquation(double alpha, double* u, double h, double endTime, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < numPoints - 1) {
        // Copie temporaire de la solution actuelle
        double tempU[3] = { u[i - 1], u[i], u[i + 1] };

        // Calcul des coefficients k1, k2, k3, k4
        double k1 = h * heatEquation(alpha, tempU, 1);
        double k2 = h * heatEquation(alpha, tempU, 1);
        double k3 = h * heatEquation(alpha, tempU, 1);
        double k4 = h * heatEquation(alpha, tempU, 1);

        // Mise à jour de la solution
        u[i] = tempU[1] + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
}

int main() {
    // Paramètres
    double alpha = 0.01;  // constante de diffusion thermique
    double endTime = 0.1; // temps final

    // Itérations sur différents maillages
    std::vector<int> mesh_sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304};
    // Fichier de sortie pour les erreurs
    std::ofstream errorFile("error_cuda.txt");
    // Fichier de sortie pour les temps d'exécution
    std::ofstream timeFile("time_cuda.txt");

    for (int mesh_size : mesh_sizes) {
        // Calcul de la solution exacte
        std::vector<double> exact_solution(mesh_size, 0.0);
        for (int i = 0; i < mesh_size; ++i) {
            double x = i * (1.0 / static_cast<double>(mesh_size));
            exact_solution[i] = exactSolution(alpha, x, endTime);
        }

        // Pas de discrétisation en espace
        double h = 1.0 / static_cast<double>(mesh_size);

        // Conditions initiales
        std::vector<double> numerical_solution(mesh_size, 0.0);

        // Initialiser la condition initiale (une gaussienne centrée)
        for (int i = 0; i < mesh_size; ++i) {
            double x = i * h;
            numerical_solution[i] = exp(-x * x / (4.0 * alpha));
        }

        // Transfert des données vers le périphérique CUDA
        double* d_numerical_solution;
        cudaMalloc(&d_numerical_solution, mesh_size * sizeof(double));
        cudaMemcpy(d_numerical_solution, numerical_solution.data(), mesh_size * sizeof(double), cudaMemcpyHostToDevice);

        // Chronométrage du kernel CUDA
        auto start = std::chrono::high_resolution_clock::now();

        // Résolution de l'équation de la chaleur avec RK4 en parallèle avec CUDA
        for (double t = 0.0; t < endTime; t += h) {
            rungeKuttaHeatEquation<<<(mesh_size + 63) / 64, 64>>>(alpha, d_numerical_solution, h, endTime, mesh_size);
            cudaDeviceSynchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Transfert des résultats depuis le périphérique CUDA
        cudaMemcpy(numerical_solution.data(), d_numerical_solution, mesh_size * sizeof(double), cudaMemcpyDeviceToHost);

        // Libération de la mémoire du périphérique CUDA
        cudaFree(d_numerical_solution);

        // Calcul de l'erreur relative moyenne
        double total_error = 0.0;
        for (int i = 0; i < mesh_size; ++i) {
            total_error += std::abs(numerical_solution[i] - exact_solution[i]);
        }
        double mean_relative_error = total_error / static_cast<double>(mesh_size);

        // Écriture des résultats dans les fichiers de sortie
        errorFile << mesh_size << " " << mean_relative_error << std::endl;
        timeFile << mesh_size << " " << duration.count() << std::endl;
    }

    // Fermeture des fichiers de sortie
    errorFile.close();
    timeFile.close();

    return 0;
}
