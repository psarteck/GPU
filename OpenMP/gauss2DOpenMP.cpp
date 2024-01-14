#include <iostream>
#include "Eigen/Dense"
#include "omp.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string>


using namespace Eigen;
using namespace std;

double f(double x, double y) {
    return x * y; 
}
double f2(double x, double y){
    return cos(x)*sin(2*y);
}

double f3(double x, double y){
    return x*y*cos(x)*sin(2*y);
}

double gauss2DIntegration(double a1, double b1, double a2, double b2, int numPoints, double (*func)(double,double)) {

    double result = 0.0;
    double weights = 2.0 / numPoints;
    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < numPoints; ++i) {
        // double xi = -1.0 + 2.0 * (double(i) + 0.5) / numPoints;
        for (int j = 0; j < numPoints; ++j) {
            // double xj = -1.0 + 2.0 * (double(j) + 0.5) / numPoints;
            result += weights * weights * func(((-1.0 + 2.0 * (double(i) + 0.5) / numPoints) + 1) / 2.0 * (b1 - a1) + a1,
                                                   ((-1.0 + 2.0 * (double(j) + 0.5) / numPoints) + 1) / 2.0 * (b2 - a2) + a2);
        }
    }

    return result * 0.25 * (b1 - a1) * (b2 - a2);
}

int main(int argc, char * argv[]) {

    int numPoints = (argc > 1) ? std::stoi(argv[1]) : 5000;

    int numThreads = (argc > 2) ? std::stoi(argv[2]) : 4;

    
    double a1 = 0.0, b1 = 10.0;
    double a2 = 0.0, b2 = 10.0;

    omp_set_num_threads(numThreads);

    double startTime = omp_get_wtime();

    double result = gauss2DIntegration(a1, b1, a2, b2, numPoints, &f3);

    double duration = omp_get_wtime() - startTime;

    double ex = 13.1913267088667;
    double error = abs(result - ex);

    std::string filename = "../Results/gauss_Op_MP_nbProc_" + std::to_string(numThreads) + ".txt";

    std::ofstream outFile(filename, std::ios_base::app);

    if (outFile.is_open()) {

        outFile << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;

        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
