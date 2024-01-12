#include <iostream>
#include "eigen-3.4.0\eigen-3.4.0\Eigen\Dense"
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

#include <curand_kernel.h>

using namespace Eigen;
using namespace std;
using namespace std::chrono;

__device__ double f(double x, double y) {
    return x * y * cos(x) * sin(2 * y);
}

__global__ void computeGauss2DPointsWeights(double* points, double* weights, int numPoints, int totalPoints) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < totalPoints) {
        int i = tid / numPoints;
        int j = tid % numPoints;

        double xi = -1.0 + 2.0 * (i + 0.5) / numPoints;
        double eta = -1.0 + 2.0 * (j + 0.5) / numPoints;
        double wi = 2.0 / numPoints;

        points[tid * 2] = xi;
        points[tid * 2 + 1] = eta;
        weights[tid] = wi;
    }
}

__global__ void gauss2DIntegrationKernel(double* result, double* points, double* weights, double a1, double b1, double a2, double b2, int numPoints, int totalPoints) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < totalPoints) {
        int i = tid / numPoints;
        int j = tid % numPoints;

        double xi = points[tid * 2];
        double eta = points[tid * 2 + 1];

        result[tid] = weights[i] * weights[j] * f((xi + 1) / 2.0 * (b1 - a1) + a1, (eta + 1) / 2.0 * (b2 - a2) + a2);
    }
}

double gauss2DIntegrationCUDA(double a1, double b1, double a2, double b2, int numPoints) {
    int totalPoints = numPoints * numPoints;

    // Allocate device memory
    double* d_points;
    double* d_weights;
    double* d_result;

    cudaMalloc((void**)&d_points, totalPoints * 2 * sizeof(double));
    cudaMalloc((void**)&d_weights, totalPoints * sizeof(double));
    cudaMalloc((void**)&d_result, totalPoints * sizeof(double));

    // Launch CUDA kernel to compute points and weights
    int threadsPerBlock = 1024;
    int numBlocks = (totalPoints + threadsPerBlock - 1) / threadsPerBlock;

    computeGauss2DPointsWeights<<<numBlocks, threadsPerBlock>>>(d_points, d_weights, numPoints, totalPoints);
    cudaDeviceSynchronize();

    // Launch CUDA kernel for integration
    gauss2DIntegrationKernel<<<numBlocks, threadsPerBlock>>>(d_result, d_points, d_weights, a1, b1, a2, b2, numPoints, totalPoints);
    cudaDeviceSynchronize();

    // Copy results from device to host
    double* h_result = new double[totalPoints];
    cudaMemcpy(h_result, d_result, totalPoints * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate the final result on the CPU
    double final_result = 0.0;
    for (int i = 0; i < totalPoints; ++i) {
        final_result += h_result[i];
    }

    // Free allocated memory
    delete[] h_result;
    cudaFree(d_points);
    cudaFree(d_weights);
    cudaFree(d_result);

    // Calculate the average and integrate over the area
    double area = (b1 - a1) * (b2 - a2);
    double integral = final_result * 0.25 * area;

    return integral;
}

void performComputation(int numPoints) {
    double a1 = 0.0, b1 = 10.0;
    double a2 = 0.0, b2 = 10.0;

    auto start_time = high_resolution_clock::now();
    double result = gauss2DIntegrationCUDA(a1, b1, a2, b2, numPoints);
    auto end_time = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

    double ex = 13.1913267088667;
    double error = abs(result - ex);

    std::string errorFilename = "error_cuda.txt";
    std::string timeFilename = "time_cuda.txt";

    std::ofstream errorFile(errorFilename, std::ios_base::app);  
    std::ofstream timeFile(timeFilename, std::ios_base::app);    

    if (errorFile.is_open() && timeFile.is_open()) {
        errorFile << std::setprecision(20) << numPoints << " " << error << std::endl;
        timeFile << std::setprecision(20) << numPoints << " " << duration << std::endl;

        errorFile.close();
        timeFile.close();
    } else {
        std::cerr << "Error: Unable to open files for writing." << std::endl;
    }
}

int main() {
    int maxExponent = 14; // 2^30

    for (int exp = 1; exp <= maxExponent; ++exp) {
        int numPoints = pow(2, exp);

        performComputation(numPoints);
    }

    return 0;
}
