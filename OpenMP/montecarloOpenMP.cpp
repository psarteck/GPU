#include <iostream>
#include <random>
#include <functional>
#include "omp.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string>
#include <fstream>

double func(double x, double y) {
    return x * y;
}

double f3(double x, double y) {
    return x * y * cos(x) * sin(2 * y);
}

//same seed per thread
/*double monteCarlo2DIntegration(std::function<double(double, double)> func, double x_min, double x_max, double y_min, double y_max, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib_x(x_min, x_max);
    std::uniform_real_distribution<double> distrib_y(y_min, y_max);

    double total = 0.0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < num_samples; ++i) {
        double x = distrib_x(gen);
        double y = distrib_y(gen);
        total += func(x, y);  // Evaluate the function at random (x, y) points
    }

    double area = (x_max - x_min) * (y_max - y_min);
    double average = total / num_samples;
    double integral = area * average;

    return integral;
}*/

double monteCarlo2DIntegration(std::function<double(double, double)> func, double x_min, double x_max, double y_min, double y_max, int num_samples) {
    double total = 0.0;

    #pragma omp parallel reduction(+:total)
    {
        int thread_id = omp_get_thread_num();
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<double> distrib_x(x_min, x_max);
        std::uniform_real_distribution<double> distrib_y(y_min, y_max);

        double local_total = 0.0;

        #pragma omp for
        for (int i = 0; i < num_samples; ++i) {
            double x = distrib_x(gen);
            double y = distrib_y(gen);
            local_total += f3(x, y);
        }

        total += local_total;
    }

    double area = (x_max - x_min) * (y_max - y_min);
    double average = total / (num_samples);
    double integral = area * average;

    return integral;
}





int main(int argc, char * argv[]) {

    double x_min = 0.0, x_max = 10.0;
    double y_min = 0.0, y_max = 10.0;

    int numPoints = (argc > 1) ? std::stoi(argv[1]) : 5000;

    int numThreads = (argc > 2) ? std::stoi(argv[2]) : 4;

    omp_set_num_threads(numThreads);

    double startTime = omp_get_wtime();

    double result = monteCarlo2DIntegration(f3, x_min, x_max, y_min, y_max, numPoints);

    double duration = omp_get_wtime() - startTime;


    // double ex = 0.25;
    double ex = 13.1913267088667;
    double error = abs(result - ex)/abs(ex);

    std::cout << std::setprecision(20) << "Result: " << result << std::endl;
    std::cout << std::setprecision(20) << "Runtime: " << duration << " seconds" << std::endl;
    std::cout << std::setprecision(20) << "Error: " << error << std::endl;


    std::string filename = "../Results/montecarlo_Op_MP_nbProc_" + std::to_string(numThreads) + ".txt";
    std::cout << filename << std::endl;

    std::ofstream outFile(filename, std::ios_base::app);

    if (outFile.is_open()) {
        outFile << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;

        outFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
    }

    return 0;
}

