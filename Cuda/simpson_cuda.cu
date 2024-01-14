#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>

constexpr int num_threads_per_block = 64;

__device__ double function(double x) {
    return 4.0 / (1.0 + x * x);
}

__global__ void integrate(double* result, int num_subintervals, double dx) {
    __shared__ volatile double partial_sum[num_threads_per_block];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    double local_sum = 0.0;
    double compensation = 0.0;
    for (int i = global_tid; i < num_subintervals; i += blockDim.x * gridDim.x) {
        double x0 = i * dx;
        double x1 = (i + 1) * dx;
        double x_mid = (x0 + x1) / 2.0;
        double f0 = function(x0);
        double f1 = function(x1);
        double f_mid = function(x_mid);
        
        double sum = f0 + 4.0 * f_mid + f1 - compensation;
        double t = local_sum + sum;
        compensation = (t - local_sum) - sum;
        local_sum = t;
    }

    partial_sum[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Store the result in global memory
    if (tid == 0) {
        result[blockIdx.x] = partial_sum[0] * dx / 6.0;
    }
}

int main() {
    const double a = 0.0;
    const double b = 1.0;

    std::ofstream output_file("../Results/output_simp_cuda.txt");

    for (int num_subintervals = 16; num_subintervals <= 1073741825; num_subintervals *= 2) {

        int num_blocks = (num_subintervals + num_threads_per_block - 1) / num_threads_per_block;

        double* result_cpu = new double[num_blocks];
        double* result_gpu;
        cudaMalloc((void**)&result_gpu, num_blocks * sizeof(double));

        auto start_time = std::chrono::high_resolution_clock::now();

        integrate<<<num_blocks, num_threads_per_block>>>(result_gpu, num_subintervals, (b - a) / num_subintervals);

        cudaMemcpy(result_cpu, result_gpu, num_blocks * sizeof(double), cudaMemcpyDeviceToHost);

        double final_result = 0.0;
        for (int i = 0; i < num_blocks; ++i) {
            final_result += result_cpu[i];
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double execution_time = duration.count();

        double pi = 3.141592653589793238462643383279502884197;
        double error = std::abs(final_result - pi);

        output_file << num_subintervals << " " << error << " " << execution_time << std::endl;

        delete[] result_cpu;
        cudaFree(result_gpu);
    }

    output_file.close();

    return 0;
}
