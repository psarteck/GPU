#include <iostream>
#include <random>
#include <functional>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>

double func(double x, double y) {
    return x * y;
}

double monteCarlo2DIntegration(std::function<double(double, double)> func, double x_min, double x_max, double y_min, double y_max, int num_samples) {
    double total = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib_x(x_min, x_max);
    std::uniform_real_distribution<double> distrib_y(y_min, y_max);

    for (int i = 0; i < num_samples; ++i) {
        double x = distrib_x(gen);
        double y = distrib_y(gen);
        total += func(x, y);
    }

    double area = (x_max - x_min) * (y_max - y_min);
    double average = total / num_samples;
    double integral = area * average;

    return integral;
}

void performComputation(double x_min, double x_max, double y_min, double y_max, long int n, std::ofstream& output_file) {

    auto start_time = std::chrono::high_resolution_clock::now();
    double result = monteCarlo2DIntegration(func, x_min, x_max, y_min, y_max, n);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    double exact = 0.25;
    double error = std::abs(result - exact) / exact;

    output_file << std::setprecision(20) << n << " " << error << " " << duration << std::endl;
}

int main(int argc, char* argv[]) {
    std::string outputFilename = "../Results/output_montecarlo_seq.txt";
    std::ofstream outputFile(outputFilename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open the output file for writing." << std::endl;
        return 1;
    }

    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    int maxExponent = 25;
    for (int exp = 4; exp <= maxExponent; ++exp) {
        long int n = pow(2, exp);
        performComputation(x_min, x_max, y_min, y_max, n, outputFile);
    }

    outputFile.close();

    return 0;
}
