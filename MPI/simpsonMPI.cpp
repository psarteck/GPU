#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <mpi.h>

using namespace std;

double function(double theta) {
    return sin(2 * theta) / ((sin(theta) + cos(theta)) * (sin(theta) + cos(theta)));
}

double function2(double x) {
    return 4.0 / (1.0 + x * x);
}

double compositeSimpsons(double a, double b, int n, double (*func)(double)) {
    double h = (b - a) / double(n);
    double integral = func(a) + func(b);

    for (int i = 1; i < n; i += 2) {
        double x = a + i * h;
        integral += 4.0 * func(x);
    }

    for (int i = 2; i < n - 1; i += 2) {
        double x = a + i * h;
        integral += 2.0 * func(x);
    }

    return integral * h / 3.0;
}

int main(int argc, char** argv) {

    int n = (argc > 1) ? std::stoi(argv[1]) : 10000;

    MPI_Init(0,0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double pi = 3.14159265358979323846;
    double a = 0.0;
    double b = 1.0;

    int remainder = n % size;
    int local_n = n / size;
    int extra_points = rank < remainder ? 1 : 0;

    local_n += extra_points;

    double local_a = a + rank * (b - a) / size;
    double local_b = local_a + (b - a) / size;


    cout << "Proc = " << rank << ", local_a = " << local_a << ", local_b = " << local_b << endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    double local_result = compositeSimpsons(local_a, local_b, local_n, &function2);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;

    double result;
    MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        std::string filename = "../Results/simp_MPI_nbProc_" + std::to_string(size) + ".txt";
        double exact = M_PI;
        double error = std::abs(exact - result);

        std::ofstream outFile(filename, std::ios_base::app);

        if (outFile.is_open()) {
            outFile << std::setprecision(20) << n << " " << error << " " << duration << std::endl;

            outFile.close();
        } else {
            std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << " pour écriture." << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}
