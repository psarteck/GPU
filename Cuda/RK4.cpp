#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>

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

    for (double t = 0; t < endTime; t += h) {
        // Copie temporaire de la solution actuelle
        std::vector<double> tempU(u);

        // Calcul des coefficients k1, k2, k3, k4
        for (int i = 1; i < numPoints - 1; ++i) {
            double k1 = h * heatEquation(alpha, tempU, i);
            double k2 = h * heatEquation(alpha, tempU, i + 0.5*h);
            double k3 = h * heatEquation(alpha, tempU, i + 0.5*h);
            double k4 = h * heatEquation(alpha, tempU, i + h);

            // Mise à jour de la solution
            u[i] = tempU[i] + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
}


int main() {
    // Iterate over different mesh sizes
    for (int expo = 1; expo <= 30; ++expo) {
        int mesh_size = pow(2, expo);

        // Parameters
        double alpha = 0.01;  // thermal diffusion constant
        double endTime = 0.1; // final time

        // Error calculation
        double error;

        // Exact solution calculation
        std::vector<double> exact_solution(mesh_size, 0.0);
        for (int i = 0; i < mesh_size; ++i) {
            double x = i * (1.0 / static_cast<double>(mesh_size));
            exact_solution[i] = exactSolution(alpha, x, endTime);
        }

        // Discretization step in space
        double h = 1.0 / static_cast<double>(mesh_size);

        // Initial conditions
        std::vector<double> numerical_solution(mesh_size, 0.0);

        auto start_time = std::chrono::high_resolution_clock::now();

        // Initialize the initial condition (a centered Gaussian)
        for (int i = 0; i < mesh_size; ++i) {
            double x = i * h;
            numerical_solution[i] = exp(-x * x / (4.0 * alpha));
        }

        // Solve the heat equation with RK4
        rungeKuttaHeatEquation(alpha, numerical_solution, h, endTime);

        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

        // Calculate the mean relative error
        double total_error = 0.0;
        for (int i = 0; i < mesh_size; ++i) {
            total_error += std::abs(numerical_solution[i] - exact_solution[i]);
        }
        double mean_relative_error = total_error / static_cast<double>(mesh_size);

        std::string errorFilename = "error.txt";
        std::string timeFilename = "time.txt";

        std::ofstream errorFile(errorFilename, std::ios_base::app);  // Open for appending
        std::ofstream timeFile(timeFilename, std::ios_base::app);    // Open for appending

        if (errorFile.is_open() && timeFile.is_open()) {
            errorFile << std::setprecision(20) << mesh_size << " " << mean_relative_error << std::endl;
            timeFile << std::setprecision(20) << mesh_size << " " << duration << std::endl;

            errorFile.close();
            timeFile.close();
        } else {
            std::cerr << "Error: Unable to open files for writing." << std::endl;
        }
    }

    return 0;
}
