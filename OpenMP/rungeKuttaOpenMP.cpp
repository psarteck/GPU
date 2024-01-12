#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "omp.h"
#include <iomanip>
#include <fstream>

// Fonction représentant la solution exacte de l'équation de la chaleur
double exactSolution(double alpha, double x, double t) {
    return 1.0 / sqrt(4.0 * M_PI * alpha * t) * exp(-x * x / (4.0 * alpha * t));
}

// Fonction représentant l'équation de la chaleur
double heatEquation(double alpha, const std::vector<double>& u, int i) {
    return alpha * (u[i - 1] - 2 * u[i] + u[i + 1]);
}

// Méthode de Runge-Kutta d'ordre 4 pour résoudre l'équation de la chaleur
void rungeKuttaHeatEquation(double alpha, std::vector<double>& u, double h, double endTime) {
    int numPoints = u.size();

    // Boucle temporelle parallélisée avec OpenMP
    #pragma omp parallel for
    for (int tIndex = 0; tIndex < static_cast<int>(endTime / h); ++tIndex) {
        // Copie temporaire de la solution actuelle
        std::vector<double> tempU(u);

        // Calcul des coefficients k1, k2, k3, k4
        for (int i = 1; i < numPoints - 1; ++i) {
            double k1 = h * heatEquation(alpha, tempU, i);
            double k2 = h * heatEquation(alpha, tempU, i + 0.5 * h);
            double k3 = h * heatEquation(alpha, tempU, i + 0.5 * h);
            double k4 = h * heatEquation(alpha, tempU, i + h);

            // Mise à jour de la solution
            u[i] = tempU[i] + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
}


int main(int argc, char *argv[]) {

    int numThreads = (argc > 2) ? std::stoi(argv[2]) : 4;

    int mesh_size = (argc > 1) ? std::stoi(argv[1]) : 5000;

    omp_set_num_threads(numThreads);

    // Paramètres
    double alpha = 0.01;  // constante de diffusion thermique
    double endTime = 0.1; // temps final

    // Itérations sur différents maillages
    double error;

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

    double startTime = omp_get_wtime();

    // Initialiser la condition initiale (une gaussienne centrée)
    for (int i = 0; i < mesh_size; ++i) {
        double x = i * h;
        numerical_solution[i] = exp(-x * x / (4.0 * alpha));
    }

    // Résolution de l'équation de la chaleur avec RK4
    rungeKuttaHeatEquation(alpha, numerical_solution, h, endTime);

    double duration = omp_get_wtime() - startTime;

    // Calcul de l'erreur relative moyenne
    double total_error = 0.0;
    for (int i = 0; i < mesh_size; ++i) {
        total_error += std::abs(numerical_solution[i] - exact_solution[i]);
    }
    double mean_relative_error = total_error / static_cast<double>(mesh_size);


    // Exporter les résultats dans un fichier CSV
    std::string filename = "../Results/RK_Op_MP_nbProc_" + std::to_string(numThreads) + ".txt";
    std::ofstream outFile(filename, std::ios_base::app);

    // Vérifier si le fichier est ouvert avec succès
    if (outFile.is_open()) {
        // Écrire la donnée dans le fichier
        outFile << std::setprecision(20) << mesh_size << " " << mean_relative_error << " " << duration << std::endl;

        // Fermer le fichier
        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
