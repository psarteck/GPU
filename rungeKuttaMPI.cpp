#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <mpi.h>

// Fonction représentant la solution exacte de l'équation de la chaleur
double exactSolution(double alpha, double x, double t) {
    return 1.0 / sqrt(4.0 * M_PI * alpha * t) * exp(-x * x / (4.0 * alpha * t));
}

// Fonction représentant l'équation de la chaleur
double heatEquation(double alpha, const std::vector<double>& u, int i) {
    return alpha * (u[i - 1] - 2 * u[i] + u[i + 1]);
}

// Méthode de Runge-Kutta d'ordre 4 pour résoudre l'équation de la chaleur
void rungeKuttaHeatEquation(double alpha, std::vector<double>& u, double h, double endTime, int rank, int size) {
    int numPoints = u.size();
    int local_size = numPoints / size;
    int start = rank * local_size;
    int end = (rank == size - 1) ? numPoints : (rank + 1) * local_size;

    // Boucle temporelle
    for (double t = 0.0; t < endTime; t += h) {
        // Copie temporaire de la solution actuelle
        std::vector<double> tempU(u);

        // Échange des bords entre les processus MPI
        if (rank > 0) {
            MPI_Sendrecv(&u[start], 1, MPI_DOUBLE, rank - 1, 0,
                         &u[start - 1], 1, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (rank < size - 1) {
            MPI_Sendrecv(&u[end - 1], 1, MPI_DOUBLE, rank + 1, 0,
                         &u[end], 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Calcul des coefficients k1, k2, k3, k4
        for (int i = start + 1; i < end - 1; ++i) {
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
void exportErrorToCSV(const std::vector<double>& errors, const std::vector<int> mesh_size, const std::string& filename, const std::vector<double> duration) {
    std::ofstream file(filename);

    if (file.is_open()) {
        int i = 0;
        for (double error : errors) {
            file << std::setprecision(20) << mesh_size[i] << " " << error << " " << duration[i] << "\n";
            i++;
        }

        file.close();
        std::cout << "L'erreur relative moyenne a été exportée dans : " << filename << std::endl;
    } else {
        std::cerr << "Erreur lors de l'ouverture du fichier CSV." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialisation de MPI
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Paramètres
    double alpha = 0.01;  // constante de diffusion thermique
    double endTime = 0.1; // temps final


    int mesh_size = (argc > 1) ? std::stoi(argv[1]) : 5000;

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

    double startTime = MPI_Wtime();

    // Résolution de l'équation de la chaleur avec RK4
    rungeKuttaHeatEquation(alpha, numerical_solution, h, endTime, rank, size);

    // Collecter les erreurs de tous les processus MPI
    std::vector<double> all_errors(mesh_size);

    MPI_Gather(&numerical_solution[rank * (mesh_size / size)], mesh_size / size, MPI_DOUBLE,
            &all_errors[rank * (mesh_size / size)], mesh_size / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double duration = MPI_Wtime() - startTime;

    // Processus MPI 0 calcule l'erreur totale
    if (rank == 0) {
        double total_error = 0.0;
        for (int i = 0; i < mesh_size; ++i) {
            total_error += std::abs(all_errors[i] - exact_solution[i]);
        }
        double mean_relative_error = total_error / static_cast<double>(mesh_size);

        std::string filename = "Results/error_mesh_MPI_nbProc_" + std::to_string(size) +".txt";
        // Exporter les résultats dans un fichier CSV
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
    }

    // Finalisation de MPI
    MPI_Finalize();

    return 0;
}
