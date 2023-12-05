#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>


using namespace Eigen;
using namespace std;

double f(double x, double y) {
    return x * y;
}

double f2(double x, double y) {
    return cos(x) * sin(2 * y);
}

double f3(double x, double y) {
    return x * y * cos(x) * sin(2 * y);
}

void computeGauss2DPointsWeights(MatrixXd &points, MatrixXd &weights, int numPoints) {
    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            double xi = -1.0 + 2.0 * (i + 0.5) / numPoints;
            double eta = -1.0 + 2.0 * (j + 0.5) / numPoints;
            double wi = 2.0 / numPoints;

            points(i, 0) = xi;
            points(j, 1) = eta;
            weights(i) = wi;
        }
    }
}

double gauss2DIntegration(double a1, double b1, double a2, double b2, int numPoints, double (*func)(double, double)) {
    MatrixXd points(numPoints, 2);
    MatrixXd weights(numPoints, 1);

    computeGauss2DPointsWeights(points, weights, numPoints);

    double result = 0.0;

    for (int i = 0; i < numPoints; ++i) {
        for (int j = 0; j < numPoints; ++j) {
            // cout <<(points(i, 0) + 1) / 2.0 * (b1 - a1) + a1 << " , " <<(points(j, 1) + 1) / 2.0 * (b2 - a2) + a2 << endl;
            result += weights(i) * weights(j) * func((points(i, 0) + 1) / 2.0 * (b1 - a1) + a1,
                                                     (points(j, 1) + 1) / 2.0 * (b2 - a2) + a2);
        }
    }

    return result * 0.25 * (b1 - a1) * (b2 - a2);
}


int main(int argc, char *argv[]) {

    int numPoints = (argc > 1) ? std::stoi(argv[1]) : 10;

    double x1 = 0.0, x2 = 10.0;
    double y1 = 0.0, y2 = 10.0;

    MPI_Init(0, 0);

    int numProcesses, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    int localNumPoints; 

    while (numPoints % numProcesses != 0){
        numPoints ++;
    }
    localNumPoints = numPoints / numProcesses;


    double local_x1, local_x2;
    double local_y1, local_y2;

    if (x2-x1 >= y2-y1){
        local_x1 = x1 + processRank * (x2-x1) / numProcesses;
        local_x2 = local_x1 + (x2-x1) / numProcesses;;
        local_y1 = y1;
        local_y2 = y2;
    }else{
        local_y1 = y1 + processRank * (y2-y1) / numProcesses;
        local_y2 = local_y1 + (y2-y1) / numProcesses;
        local_x1 = x1;
        local_x2 = x2;
    }

    cout << "Intervale du process " << processRank << " [" << local_x1 << "," << local_x2 << "]x[" << local_y1 << "," << local_y2 << "]" <<endl;
    double startTime = MPI_Wtime();

    double localResult = gauss2DIntegration(local_x1, local_x2, local_y1, local_y2, localNumPoints, &f3);

    double duration = MPI_Wtime() - startTime;

    double globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (processRank == 0) {

        double ex = 13.1913267088667;
        double error = abs(globalResult - ex);
        cout << endl;
        std::cout << std::setprecision(20) << "Result: " << globalResult << std::endl;
        std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;
        std::cout << std::setprecision(20) << "Error: " << error << std::endl;

        std::string filename = "Results/gauss2DMPI_nbProc_" + std::to_string(numProcesses) + ".txt";
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
    }

    MPI_Finalize();

    return 0;
}
