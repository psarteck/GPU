#include <iostream>
#include <cmath>
#include <vector>
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

    // Boucle temporelle
    for (double t = 0.0; t < endTime; t += h) {
        // Copie temporaire de la solution actuelle
        std::vector<double> tempU(u);

        // Calcul des coefficients k1, k2, k3, k4
        for (int i = 1; i < numPoints - 1; ++i) {
            double k1 = h * heatEquation(alpha, tempU, i);
            double k2 = h * heatEquation(alpha, tempU, i);
            double k3 = h * heatEquation(alpha, tempU, i);
            double k4 = h * heatEquation(alpha, tempU, i);

            // Mise à jour de la solution
            u[i] = tempU[i] + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
}

// Fonction pour exporter l'erreur relative moyenne dans un fichier CSV
void exportErrorToCSV(const std::vector<double>& errors, const std::vector<int> mesh_size, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        int i = 0 ;
        for (double error : errors) {
            file << error << " " << 1.0/mesh_size[i] << "\n";
            i++;
        }

        file.close();
        std::cout << "L'erreur relative moyenne a été exportée dans : " << filename << std::endl;
    } else {
        std::cerr << "Erreur lors de l'ouverture du fichier CSV." << std::endl;
    }
}

int main() {
    // Paramètres
    double alpha = 0.01;  // constante de diffusion thermique
    double endTime = 0.1; // temps final





    // Itérations sur différents maillages
    std::vector<int> mesh_sizes = {256, 512,1024,2048,4096,8192,16384,32768};
    std::vector<double> errors;

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

        // Résolution de l'équation de la chaleur avec RK4
        rungeKuttaHeatEquation(alpha, numerical_solution, h, endTime);

        // Calcul de l'erreur relative moyenne
        double total_error = 0.0;
        for (int i = 0; i < mesh_size; ++i) {
            total_error += std::abs(numerical_solution[i] - exact_solution[i]);
        }
        double mean_relative_error = total_error / static_cast<double>(mesh_size);

        // Ajouter l'erreur à la liste
        errors.push_back(mean_relative_error);
    }
            // Exporter les résultats dans un fichier CSV
    std::string filename = "Results/error_mesh.csv";
    exportErrorToCSV(errors, mesh_sizes, filename);
    return 0;
}
