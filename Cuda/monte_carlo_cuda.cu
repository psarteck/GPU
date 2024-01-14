#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <string>

#include <curand_kernel.h>

__global__ void monteCarlo2DIntegrationKernel(double* result, double x_min, double x_max, double y_min, double y_max, long long int num_samples, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    long long int samples_per_thread = (num_samples + total_threads - 1) / total_threads;

    seed += tid;
    curandState state;
    curand_init(seed, tid, 0, &state);

    double total = 0.0;

    for (long long int i = 0; i < samples_per_thread; ++i) {
        double x = curand_uniform_double(&state) * (x_max - x_min) + x_min;
        double y = curand_uniform_double(&state) * (y_max - y_min) + y_min;
        total += x * y * cos(x) * sin(2 * y);
    }

    result[tid] = total;
}

double monteCarlo2DIntegrationCUDA(double x_min, double x_max, double y_min, double y_max, long long int num_samples, int num_blocks, int threads_per_block) {

    int total_threads = num_blocks * threads_per_block;

    double* d_result;
    cudaMalloc((void**)&d_result, total_threads * sizeof(double));

    monteCarlo2DIntegrationKernel<<<num_blocks, threads_per_block>>>(d_result, x_min, x_max, y_min, y_max, num_samples, time(0));

    double* h_result = new double[total_threads];
    cudaMemcpy(h_result, d_result, total_threads * sizeof(double), cudaMemcpyDeviceToHost);

    double final_result = 0.0;
    for (int i = 0; i < total_threads; ++i) {
        final_result += h_result[i];
    }

    delete[] h_result;
    cudaFree(d_result);

    double area = (x_max - x_min) * (y_max - y_min);
    double average = final_result / num_samples;
    double integral = area * average;

    return integral;
}

void performComputation(double x_min, double x_max, double y_min, double y_max, long long int n, int num_blocks, int threads_per_block, std::ofstream& output_file) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = monteCarlo2DIntegrationCUDA(x_min, x_max, y_min, y_max, n, num_blocks, threads_per_block);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    double exact = 13.1913267088667;
    double error = std::abs(result - exact) / exact;

    output_file << std::setprecision(20) << n << " " << error << " " << duration << std::endl;
}

int main() {
    std::string outputFilename = "../Results/output_montecarlo_cuda.txt";
    std::ofstream outputFile(outputFilename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open the output file for writing." << std::endl;
        return 1;
    }

    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    int maxExponent = 25;
    int num_blocks = 3584;
    int threads_per_block = 64;

    for (int exp = 4; exp <= maxExponent; ++exp) {
        long long int n = static_cast<long long int>(std::pow(2, exp));
        performComputation(x_min, x_max, y_min, y_max, n, num_blocks, threads_per_block, outputFile);
    }

    outputFile.close();

    return 0;
}
