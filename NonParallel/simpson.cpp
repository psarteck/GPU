#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

using std::string;
using namespace std;
using namespace std::chrono;

double funcCosSin(double x) {
    return 5 * (cos(M_PI * x) * sin(2 * M_PI * x));
}

double compositeSimpsons_3_8(double a, double b, long int n, double (*func)(double)) {
    while (n % 3 != 0) {
        n++;
    }

    double h = (b - a) / double(n);
    double integral = (func(a) + func(b));

    for (int i = 1; i < n - 1; i += 3) {
        double x = a + static_cast<double>(i) * h;
        integral += 3.0 * func(x);
        x = a + static_cast<double>(i + 1) * h;
        integral += 3.0 * func(x);
    }

    for (int i = 3; i < n - 1; i += 3) {
        double x = a + double(i) * h;
        integral += 2.0 * func(x);
    }

    return 3.0 * integral * h / 8.0;
}

void performComputation(double a, double b, long int n) {
    auto start_time = high_resolution_clock::now();
    double result = compositeSimpsons_3_8(a, b, n, &funcCosSin);
    auto end_time = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

    double val_exacte = 20 / (3 * M_PI);

    double error = abs(result - val_exacte);

    std::string errorFilename = "../Results/error.txt";
    std::string timeFilename = "../Results/time.txt";

    std::ofstream errorFile(errorFilename, std::ios_base::app);
    std::ofstream timeFile(timeFilename, std::ios_base::app);

    if (errorFile.is_open() && timeFile.is_open()) {
        errorFile << std::setprecision(20) << n << " " << error << std::endl;
        timeFile << std::setprecision(20) << n << " " << duration << std::endl;

        errorFile.close();
        timeFile.close();
    } else {
        std::cerr << "Erreur : Impossible d'ouvrir les fichiers pour écriture." << std::endl;
    }
}

int main(int argc, char* argv[]) {

    double a = 0;
    double b = 1;

    int maxExponent = 30; // 2^30
    for (int exp = 1; exp <= maxExponent; ++exp) {
        long int n = pow(2, exp);

        performComputation(a, b, n);
    }

    return 0;
}

