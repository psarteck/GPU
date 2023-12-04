#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

// Function to integrate (modify as needed)
double function(double x) {
    return 4.0 / (1.0 + x * x);
}

// Composite Simpson's rule for numerical integration
double compositeSimpsons(double a, double b, int n) {
    double h = (b - a) / double(n);
    double integral = function(a) + function(b);

    for (int i = 1; i < n; i += 2) {
        double x = a + i * h;
        integral += 4.0 * function(x);
    }
    for (int i = 2; i < n - 1; i += 2) {
        double x = a + i * h;
        integral += 2.0 * function(x);
    }
    return integral * h / 3.0;
}

int main() {
    // Define integration interval [a, b]
    double a = 0.0;
    double b = 1.0;

    // Number of sub-intervals
    int n = 100000;

    // Measure runtime
    auto startTime = std::chrono::high_resolution_clock::now();

    // Calculate the integral using composite Simpson's rule
    double result = compositeSimpsons(a, b, n);

    // Measure end time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;

    double pi = 3.14159265358979323846;
    // Output the result and runtime
    std::cout << std::setprecision(20) << "Pi: " <<  pi << std::endl;
    std::cout << std::setprecision(20) << "Result: " <<  result-pi << std::endl;
    std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;
    // printf("Result : %lf\n", result<double>);

    const char* filename = "data.txt";
    
    // Ouvrir le fichier en mode écriture
    std::ofstream outFile(filename);

    // Vérifier si le fichier est ouvert avec succès
    if (outFile.is_open()) {
        // Écrire la donnée dans le fichier
        outFile << result << " " << duration << std::endl;

        // Fermer le fichier
        outFile.close();

        std::cout << "Donnée écrite avec succès dans " << filename << std::endl;
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
