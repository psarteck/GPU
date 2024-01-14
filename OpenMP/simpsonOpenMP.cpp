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

double function(double x) {
    return 4.0 / (1.0 + x * x);
}

double trigo(double theta){
    return sin(2.0*theta) / ((sin(theta) + cos(theta))*(sin(theta) + cos(theta))) ;
}

double compositeSimpsons_3_8(double a, double b, int n, double (*func)(double)) { 

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
        double x = a + static_cast<double>(i) * h;
        integral += 2.0 * func(x); 
    }

    return 3.0 *integral * h / 8.0;
}

double compositeSimpsons(double a, double b, int  n, double (*func)(double)) { 
    double h = (b - a) / double(n);
    double integral = func(a) + func(b);

    #pragma omp parallel for reduction(+:integral)
    for (long long int  i = 1; i < n; i += 2) {
        double x = a + i * h;
        integral += 4.0 * func(x);
    }

    #pragma omp parallel for reduction(+:integral)
    for (long long int  i = 2; i < n - 1; i += 2) {
        double x = a + i * h;
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

    omp_set_num_threads(numThreads);

    double startTime = omp_get_wtime();

    double result = compositeSimpsons_3_8(a, b, n, &function);

    double endTime = omp_get_wtime();

    double duration = endTime - startTime;

    double val_exacte = M_PI;

    double error = abs(result - val_exacte);

    std::string filename = "../Results/simp_Op_MP_nbProc_" + std::to_string(numThreads) + ".txt";

    std::ofstream outFile(filename, std::ios_base::app);

    if (outFile.is_open()) {
        outFile << std::setprecision(20) << n << " " << error << " " << duration << std::endl;

        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
