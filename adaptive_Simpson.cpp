#include <iostream>
#include <cmath>
#include "omp.h"
#include <fstream>
#include <iomanip>
#include <math.h>
#include <string>
#include <fstream>
using namespace std;

// Fonction à intégrer (modifier selon les besoins)
double f(double x, double &valExact) {
    return 4 / (1 + x * x);
}

double cos1x(double x) {
    return cos(1/x);
}

double func2(double x){
    return sin(1/x) / (x*x);
}



double simpson(double (*f)(double), double a,double b){
    return (b-a)/6*(f(a)+4*f((a+b)/2)+f
     (b));
}

double int_adaptsimpson(double (*f)(double), double a, double b, double tau){
    double m=(a+b)/2;
    double Sab=simpson(f,a,b),Sam=simpson(f,a,m),Smb=simpson(f,m,b);
    if (fabs(Sab-Sam-Smb)/15<tau)
        return Sam+Smb+(Sam+Smb-Sab)/15;
    else
        return int_adaptsimpson(f,a,m,tau)+int_adaptsimpson(f,m,b,tau);
}

int main() {
    
    double a = 1/(2*M_PI);
    double b = 1/M_PI;
    double epsilon = 1e-15; // Ajustez le seuil d'erreur

    double result = int_adaptsimpson(&func2, a, b, epsilon);

    
    double valExact = -2.0;//M_PI;

    double error = result - valExact;
    cout << std::setprecision(20) << "Résultat de l'intégration adaptative : " << result << " | " << error << endl;

    return 0;
}
