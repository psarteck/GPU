#include <iostream>
#include <cmath>
#include "omp.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string>
#include <fstream>

using std::string;
using namespace std;

double funcCosSin(double x){
    return 5*(cos(M_PI*x)*sin(2*M_PI*x));
}

double cos1x(double x){
    return cos(1/x);
}

// trigo to integrate (modify as needed)
double function(double x) {
    return 4.0 / (1.0 + x * x);
}

double trigo(double theta){
    return sin(2.0*theta) / ((sin(theta) + cos(theta))*(sin(theta) + cos(theta))) ;
}

double compositeSimpsons_3_8(double a, double b, long int n, double (*func)(double)) { 

    while (n % 3 != 0){
        n++;
    }

    double h = (b - a) / double(n);
    double integral = (func(a) + func(b));

    #pragma omp parallel for reduction(+:integral)
    for (int i = 1; i < n - 1 ; i+=3) {
        double x = a + static_cast<double>(i) * h;
        integral += 3.0 * func(x);
        x = a + static_cast<double>(i+1) * h;
        integral += 3.0 * func(x);
    }

    #pragma omp parallel for reduction(+:integral)
    for (int i = 3; i < n - 1; i += 3) {
        double x = a + double(i) * h;
        integral += 2.0 * func(x); 
    }

    return 3.0 *integral * h / 8.0;
}

// Composite Simpson's rule for numerical integration
double compositeSimpsons(double a, double b, long int n, double (*func)(double)) { 
    double h = (b - a) / double(n);
    double integral = func(a) + func(b);

    #pragma omp parallel for reduction(+:integral)
    for (int i = 1; i < n; i += 2) {
        double x = a + double(i) * h;
        integral += 4.0 * func(x);
        cout << i << endl;
    }

    #pragma omp parallel for reduction(+:integral)
    for (int i = 2; i < n - 1; i += 2) {
        double x = a + double(i) * h;
        integral += 2.0 * func(x);
    }

    return integral * h / 3.0;
}

int main(int argc, char * argv[]) {

    
    // Define integration interval [a, b]
    double a = 0;
    double b = 1;//M_PI/2.0;

    // Number of sub-intervals
    // int n = std::stoi(argv[1]);

    int n = (argc > 1) ? std::stoi(argv[1]) : 10000;

    int numThreads = (argc > 2) ? std::stoi(argv[2]) : 4;

    // Set the number of threads
    omp_set_num_threads(numThreads);

    // Measure runtime
    double startTime = omp_get_wtime();

    // Calculate the integral using composite Simpson's rule
    double result = compositeSimpsons_3_8(a, b, n, &funcCosSin);

    

    // Measure end time
    double endTime = omp_get_wtime();

    double duration = endTime - startTime;

    double val_exacte = 20/(3*M_PI);
    double val_exacte2 = 2.122065907891937810;

    cout << std::setprecision(25)<< val_exacte << endl;
    cout << std::setprecision(25)<< val_exacte2 << endl;


    double error = abs(result - val_exacte);
    // Output the result and runtime
    std::cout << std::setprecision(20) << "Result: " << result << std::endl;
    std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;
    std::cout << std::setprecision(20) << "Error: " << error << std::endl;


    std::string filename = "data_nbProc_" + std::to_string(numThreads) + ".txt";
    std::cout << filename << std::endl;

    // Ouvrir le fichier en mode écriture
    std::ofstream outFile(filename, std::ios_base::app);

    // Vérifier si le fichier est ouvert avec succès
    if (outFile.is_open()) {
        // Écrire la donnée dans le fichier
        outFile << std::setprecision(20) << n << " " << error << " " << duration << std::endl;

        // Fermer le fichier
        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
