#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <mpi.h>

double func(double x, double y) {
    return x * y;  // Example function: x * y
}

double monteCarlo2DIntegration(int localNumPoints, double x_min, double x_max, double y_min, double y_max, std::mt19937 &gen) {
    std::uniform_real_distribution<double> distrib_x(x_min, x_max);
    std::uniform_real_distribution<double> distrib_y(y_min, y_max);

    double localTotal = 0.0;
    for (int i = 0; i < localNumPoints; ++i) {
        double x = distrib_x(gen);
        double y = distrib_y(gen);
        localTotal += func(x, y);
    }

    return localTotal;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int numPoints = (argc > 1) ? std::stoi(argv[1]) : 1000000; // Default 1 million points

    int numProcesses, processRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    int localNumPoints = numPoints / numProcesses;
    int remainingPoints = numPoints % numProcesses;

    // Distribute the remaining points among first 'remainingPoints' processes
    int myPoints = (processRank < remainingPoints) ? localNumPoints + 1 : localNumPoints;

    // Seed random number generator differently for each process
    std::random_device rd;
    std::mt19937 gen(rd() + processRank);

    double startTime = MPI_Wtime();

    double localResult = monteCarlo2DIntegration(myPoints, 0.0, 1.0, 0.0, 1.0, gen);

    double globalResult;
    MPI_Reduce(&localResult, &globalResult, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double endTime = MPI_Wtime();
    double duration = endTime - startTime;

    if (processRank == 0) {
        double area = (1.0 - 0.0) * (1.0 - 0.0); // Area for this particular integration (x_max - x_min) * (y_max - y_min)
        double average = globalResult / numPoints;
        double integral = area * average;

        double ex = 0.25; // Expected solution (change accordingly)
        double error = std::abs(integral - ex * area);

        std::cout << std::setprecision(20) << "Result: " << integral << std::endl;
        std::cout << std::setprecision(20) << "Error: " << error << std::endl;
        std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;

        std::string filename = "../Results/montecarlo_MPI_nbProc_" + std::to_string(numProcesses) + ".txt";
        std::ofstream outFile(filename, std::ios_base::app);
        if (outFile.is_open()) {
            outFile << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;
            outFile.close();
        } else {
            std::cerr << "Error: Unable to open the file." << std::endl;
        }
    }

    MPI_Finalize();

    return 0;
}




