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

void computeGauss2DPointsWeights(MatrixXd& points, MatrixXd& weights, int numPoints) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            double xi = -1.0 + 2.0 * (i + 0.5) / numPoints;
            double eta = -1.0 + 2.0 * (j + 0.5) / numPoints;
            double wi = 2.0 / numPoints;

            // #pragma omp critical
            // {
                points(i, 0) = xi;
                points(j, 1) = eta;
                weights(i) = wi;
            // }
        }
    }
}



double gauss2DIntegration(double a1, double b1, double a2, double b2, int numPoints, double (*func)(double,double)) {
    MatrixXd points(numPoints, 2);
    MatrixXd weights(numPoints, 1);

    computeGauss2DPointsWeights(points, weights, numPoints);

    double result = 0.0;

    #pragma omp parallel for reduction(+:result) collapse(2)
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            result += weights(i) * weights(j) * func((points(i, 0) + 1) / 2.0 * (b1 - a1) + a1,
                                                   (points(j, 1) + 1) / 2.0 * (b2 - a2) + a2);
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

    // double ex = -0.161007927143812; //F2 
    double ex = 13.1913267088667;
    double error = abs(result - ex);

    std::cout << std::setprecision(20) << "Result: " << result << std::endl;
    std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;
    std::cout << std::setprecision(20) << "Error: " << error << std::endl;


    std::string filename = "../Results/gauss_Op_MP_nbProc_" + std::to_string(numThreads) + ".txt";
    std::cout << filename << std::endl;

    // Ouvrir le fichier en mode écriture
    std::ofstream outFile(filename, std::ios_base::app);

    // Vérifier si le fichier est ouvert avec succès
    if (outFile.is_open()) {
        // Écrire la donnée dans le fichier
        outFile << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;

        // Fermer le fichier
        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}
