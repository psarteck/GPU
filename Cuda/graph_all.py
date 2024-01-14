import sys
import matplotlib.pyplot as plt
import numpy as np

def lire_donnees(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()

    x = []
    y_error = []
    y_time = []

    for ligne in lignes:
        valeurs = ligne.split()
        x.append(float(valeurs[0]))
        
        error_value = float(valeurs[1])
        if np.isinf(error_value):
            y_error.append(0)
        else:
            y_error.append(error_value)

        time_value = float(valeurs[2])
        if np.isinf(time_value):
            y_time.append(0)
        else:
            y_time.append(time_value)

    min_length = min(len(x), len(y_error), len(y_time))
    x = x[:min_length]
    y_error = y_error[:min_length]
    y_time = y_time[:min_length]

    return np.array(x), np.array(y_error), np.array(y_time)

if len(sys.argv) != 2:
    print("Usage: python script.py <label>")
    sys.exit(1)

label = sys.argv[1]

x, y_error_seq, y_time_seq = lire_donnees(f'../Results/output_{label}_seq.txt')
x, y_error_omp, y_time_omp = lire_donnees(f'../Results/{label}_Op_MP_nbProc_8.txt')
x, y_error_mpi, y_time_mpi = lire_donnees(f'../Results/{label}_MPI_nbProc_8.txt')
x, y_error_cuda, y_time_cuda = lire_donnees(f'../Results/output_{label}_cuda.txt')

plt.figure(figsize=(10, 6))
plt.loglog(x, y_time_seq, '-o',label='Séquentiel')
plt.loglog(x, y_time_omp,'-o' ,label='OpenMP')
plt.loglog(x, y_time_mpi,'-o', label='MPI')
plt.loglog(x, y_time_cuda,'-o', label='CUDA')
plt.title(f'Graphique du temps en échelle log pour la méthode {label}')
plt.xlabel('Nombre de Subdivisions')
plt.ylabel('Temps (s)')
plt.xscale('log', base=2)
plt.legend()
plt.grid(True)
plt.savefig(f'../Images/time_{label}.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog(x, y_time_seq, '-o',label='Séquentiel')
plt.loglog(x, y_time_omp,'-o' ,label='OpenMP')
plt.loglog(x, y_time_mpi,'-o', label='MPI')
plt.loglog(x, y_time_cuda,'-o', label='CUDA')
plt.title(f'Graphique de l\'erreur en échelle log pour la méthode {label}')
plt.xlabel('Nombre de Subdivisions')
plt.ylabel('Erreur')
plt.xscale('log', base=2)
plt.legend()
plt.grid(True)
plt.savefig(f'../Images/error_{label}.png')
plt.show()
