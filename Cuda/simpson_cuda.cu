#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

__device__ double function(double x) {
    return 4.0 / (1.0 + x * x);
}

__global__ void integrate(double* result, int num_subintervals, double dx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    for (int i = tid; i < num_subintervals; i += blockDim.x * gridDim.x) {
        double x0 = i * dx;
        double x1 = (i + 1) * dx;
        double x_mid = (x0 + x1) / 2.0;
        sum += function(x0) + 4.0 * function(x_mid) + function(x1);
    }
    
    result[tid] = sum * dx / 6.0;
}

int main() {
    const int num_subintervals_start = 2;
    const int num_subintervals_end = 1073741825;
    const int num_threads_per_block = 64;
    const int num_blocks = 3584;
    const double a = 0.0; 
    const double b = 1.0; 

    std::ofstream error_file("error_cuda.txt");
    std::ofstream time_file("time_cuda.txt");

    for (int num_subintervals = num_subintervals_start; num_subintervals <= num_subintervals_end; num_subintervals *= 2) {

        double* result_cpu = new double[num_subintervals];

        double* result_gpu;
        cudaMalloc((void**)&result_gpu, num_subintervals * sizeof(double));


        auto start_time = std::chrono::high_resolution_clock::now();

        integrate<<<num_blocks, num_threads_per_block>>>(result_gpu, num_subintervals, (b - a) / num_subintervals);

        cudaMemcpy(result_cpu, result_gpu, num_subintervals * sizeof(double), cudaMemcpyDeviceToHost);

        double final_result = 0.0;
        for (int i = 0; i < num_subintervals; ++i) {
            final_result += result_cpu[i];
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double execution_time = duration.count();

        double pi = 3.141592653589793238462643383279502884197;
        double error = std::abs(final_result - pi);

        error_file << num_subintervals << " " << error << std::endl;
        time_file << num_subintervals << " " << execution_time << std::endl;

        delete[] result_cpu;
        cudaFree(result_gpu);
    }

    error_file.close();
    time_file.close();

    return 0;
}
