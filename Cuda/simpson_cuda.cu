#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
// Fonction à intégrer
__device__ double function(double x) {
    return 4.0 / (1.0 + x * x);
}

// Kernel CUDA pour l'évaluation de l'intégrale avec la règle de Simpson composée
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
    const int num_subintervals_start = 1000;
    const int num_subintervals_end = 1000000000;
    const int num_threads_per_block_values[] = {128, 256, 512, 1024};
    const int num_values = sizeof(num_threads_per_block_values) / sizeof(num_threads_per_block_values[0]);
    const double a = 0.0; // Borne inférieure
    const double b = 1.0; // Borne supérieure

    std::ofstream error_file("error_results.txt");
    std::ofstream time_file("time_results.txt");

    for (int num_threads_idx = 0; num_threads_idx < num_values; ++num_threads_idx) {
        int num_threads_per_block = num_threads_per_block_values[num_threads_idx];

        error_file << "Threads per block: " << num_threads_per_block << std::endl;
        time_file << "Threads per block: " << num_threads_per_block << std::endl;

        for (int num_subintervals = num_subintervals_start; num_subintervals <= num_subintervals_end; num_subintervals *=10 ) {
            // Allocation mémoire sur le CPU pour stocker les résultats
            double* result_cpu = new double[num_subintervals];

            // Allocation mémoire sur le GPU
            double* result_gpu;
            cudaMalloc((void**)&result_gpu, num_subintervals * sizeof(double));

            // Paramètres du GPU
            const int num_blocks = std::min(30, (num_subintervals + num_threads_per_block - 1) / num_threads_per_block);

            // Mesurer le temps d'exécution
            auto start_time = std::chrono::high_resolution_clock::now();

            // Appeler le kernel CUDA pour l'évaluation de l'intégrale
            integrate<<<num_blocks, num_threads_per_block>>>(result_gpu, num_subintervals, (b - a) / num_subintervals);

            // Copier les résultats du GPU vers le CPU
            cudaMemcpy(result_cpu, result_gpu, num_subintervals * sizeof(double), cudaMemcpyDeviceToHost);

            // Calculer le résultat final en additionnant les résultats de chaque thread
            double final_result = 0.0;
            for (int i = 0; i < num_subintervals; ++i) {
                final_result += result_cpu[i];
            }

            // Mesurer le temps total d'exécution
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            double execution_time = duration.count();
			double pi = 3.141592653589793238462643383279502884197 ;
            // Calculer l'erreur par rapport à la valeur exacte
            double error = std::abs(final_result -  pi);
			//if (error == 0) {std::cout << std::setprecision(19) << final_result << std::endl ;}
            // Enregistrement des résultats dans les fichiers
            error_file << num_subintervals << " " << error << std::endl;
            time_file << num_subintervals << " " << execution_time << std::endl;

            // Libérer la mémoire
            delete[] result_cpu;
            cudaFree(result_gpu);
        }
    }

    error_file.close();
    time_file.close();

    return 0;
}
