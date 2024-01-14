#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>

using std::string;
using namespace std;
using namespace std::chrono;

double function(double x) {
    return 4.0 / (1.0 + x * x);
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

void performComputation(double a, double b, long int n, ofstream& output_file) {
    auto start_time = high_resolution_clock::now();
    double result = compositeSimpsons_3_8(a, b, n, &function);
    auto end_time = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end_time - start_time).count() / 1000.0;

    double error = abs(result - M_PI);

    if (output_file.is_open()) {
        output_file << std::setprecision(20) << n << " " << error << " " << duration << std::endl;
    } else {
        cerr << "Erreur : Impossible d'écrire dans le fichier de sortie." << endl;
    }
}

int main(int argc, char* argv[]) {

    double a = 0;
    double b = 1;

    int maxExponent = 30;
    ofstream output_file("../Results/output_simp_seq.txt");

    if (output_file.is_open()) {
        for (int exp = 1; exp <= maxExponent; ++exp) {
            long int n = pow(2, exp);
            performComputation(a, b, n, output_file);
        }

        output_file.close();
    } else {
        cerr << "Erreur : Impossible d'ouvrir le fichier de sortie pour écriture." << endl;
    }

    return 0;
}
