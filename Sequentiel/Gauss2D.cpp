#include <iostream>
#include "..\Cuda\eigen-3.4.0\eigen-3.4.0\Eigen\Dense"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

double f3(double x, double y) {
    return x * y * cos(x) * sin(2 * y);
}

void computeGauss2DPointsWeights(MatrixXd& points, MatrixXd& weights, int numPoints) {
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
            result += weights(i) * weights(j) * func((points(i, 0) + 1) / 2.0 * (b1 - a1) + a1,
                (points(j, 1) + 1) / 2.0 * (b2 - a2) + a2);
        }
    }

    return result * 0.25 * (b1 - a1) * (b2 - a2);
}

void performComputation(int numPoints, std::ofstream& output_file) {
    double a1 = 0.0, b1 = 10.0;
    double a2 = 0.0, b2 = 10.0;

    auto start_time = high_resolution_clock::now();
    double result = gauss2DIntegration(a1, b1, a2, b2, numPoints, &f3);
    auto end_time = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

    double ex = 13.1913267088667;
    double error = abs(result - ex);

    output_file << std::setprecision(20) << numPoints << " " << error << " " << duration << std::endl;
}

int main(int argc, char* argv[]) {
    std::string outputFilename = "../Results/output_gauss_seq.txt";
    std::ofstream outputFile(outputFilename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open the output file for writing." << std::endl;
        return 1;
    }

    int maxExponent = 14;

    for (int exp = 4; exp <= maxExponent; ++exp) {
        int numPoints = pow(2, exp);
        performComputation(numPoints, outputFile);
    }

    outputFile.close();

    return 0;
}
