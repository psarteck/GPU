#include <iostream>
#include "Eigen/Dense"
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

double gauss2DIntegration(double a1, double b1, double a2, double b2, int numPointsX, int numPointsY, double (*func)(double, double)) {

    double weightX = 2.0/double(numPointsX);
    double weightY = 2.0/double(numPointsY);
    double result = 0.0;

    for (int i = 0; i < numPointsX; ++i) {
        double xi = -1.0 + 2.0 * (double(i) + 0.5) / double(numPointsX);
        for (int j = 0; j < numPointsY; ++j) {
            double xj = -1.0 + 2.0 * (double(j) + 0.5) / double(numPointsY);
            result += weightX * weightY * func((xi + 1) / 2.0 * (b1 - a1) + a1,
                                                     (xj + 1) / 2.0 * (b2 - a2) + a2);
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

    localNumPoints = numPoints / numProcesses;
    int remainder = numPoints % numProcesses;


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

    double startTime = MPI_Wtime();

    double localResult;
    if (x2-x1 >= y2-y1){
        localResult = gauss2DIntegration(local_x1, local_x2, local_y1, local_y2, localNumPoints, numPoints, &f3);
    }
    else{
        localResult = gauss2DIntegration(local_x1, local_x2, local_y1, local_y2, numPoints, localNumPoints, &f3);
    }

    double duration = MPI_Wtime() - startTime;

    double globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (processRank == 0) {

        double ex = 13.1913267088667;
        double error = abs(globalResult - ex);

        std::string filename = "../Results/gauss_MPI_nbProc_" + std::to_string(numProcesses) + ".txt";

        std::ofstream outFile(filename, std::ios_base::app);

        if (outFile.is_open()) {
            outFile << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;

            outFile.close();
        } else {
            std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}
