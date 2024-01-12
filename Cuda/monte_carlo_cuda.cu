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

    // Use a different seed for each thread
    seed += tid;
    curandState state;
    curand_init(seed, tid, 0, &state);

    double total = 0.0;

    for (long long int i = 0; i < samples_per_thread; ++i) {
        double x = curand_uniform_double(&state) * (x_max - x_min) + x_min;
        double y = curand_uniform_double(&state) * (y_max - y_min) + y_min;
        total += x * y;
    }

    result[tid] = total;
}

double monteCarlo2DIntegrationCUDA(double x_min, double x_max, double y_min, double y_max, long long int num_samples, int num_blocks, int threads_per_block) {
    // Compute the total number of threads
    int total_threads = num_blocks * threads_per_block;

    // Allocate device memory for results
    double* d_result;
    cudaMalloc((void**)&d_result, total_threads * sizeof(double));

    // Launch the CUDA kernel
    monteCarlo2DIntegrationKernel<<<num_blocks, threads_per_block>>>(d_result, x_min, x_max, y_min, y_max, num_samples, time(0));

    // Copy results from device to host
    double* h_result = new double[total_threads];
    cudaMemcpy(h_result, d_result, total_threads * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate the final result on the CPU
    double final_result = 0.0;
    for (int i = 0; i < total_threads; ++i) {
        final_result += h_result[i];
    }

    // Free allocated memory
    delete[] h_result;
    cudaFree(d_result);

    // Calculate the average and integrate over the area
    double area = (x_max - x_min) * (y_max - y_min);
    double average = final_result / num_samples;
    double integral = area * average;

    return integral;
}

void performComputation(double x_min, double x_max, double y_min, double y_max, long long int n, int num_blocks, int threads_per_block) {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = monteCarlo2DIntegrationCUDA(x_min, x_max, y_min, y_max, n, num_blocks, threads_per_block);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    double exact = 0.25;
    double error = std::abs(result - exact) / exact;

    std::string errorFilename = "error_cuda.txt";
    std::string timeFilename = "time_cuda.txt";

    std::ofstream errorFile(errorFilename, std::ios_base::app);  // Open for appending
    std::ofstream timeFile(timeFilename, std::ios_base::app);    // Open for appending

    if (errorFile.is_open() && timeFile.is_open()) {
        errorFile << std::setprecision(20) << n << " " << error << std::endl;
        timeFile << std::setprecision(20) << n << " " << duration << std::endl;

        errorFile.close();
        timeFile.close();
    } else {
        std::cerr << "Error: Unable to open files for writing." << std::endl;
    }
}

int main() {
    double x_min = 0.0, x_max = 1.0;
    double y_min = 0.0, y_max = 1.0;

    int maxExponent = 39; // 2^31
    int num_blocks = 3584;
    int threads_per_block = 64;

    for (int exp = 1; exp <= maxExponent; ++exp) {
        long long int n = static_cast<long long int>(std::pow(2, exp));

        performComputation(x_min, x_max, y_min, y_max, n, num_blocks, threads_per_block);
    }

    return 0;
}
