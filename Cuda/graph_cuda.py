import matplotlib.pyplot as plt
import numpy as np


def lire_donnees(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        lignes = fichier.readlines()
    
    x = []
    y = []

    for ligne in lignes:
        valeurs = ligne.split()
        x.append(float(valeurs[0]))
        y.append(float(valeurs[1]))

    return np.array(x), np.array(y)


x_time_cuda, y_time_cuda = lire_donnees('time_cuda.txt')
x_time, y_time = lire_donnees('time.txt')
x_error, y_error = lire_donnees('error.txt')
x_error_cuda, y_error_cuda = lire_donnees('error_cuda.txt')

plt.figure(figsize=(10, 6))
plt.plot(x_time_cuda, y_time_cuda, label='CUDA')
plt.plot(x_time, y_time, label='Séquentiel')
plt.title('Graphique du temps en échelle linéaire rk4')
plt.xlabel('Nombre de Subdivision')
plt.ylabel('Temps (s)')
plt.xscale('log', base=2) 
plt.legend()
plt.grid(True)
plt.savefig('time.png') 
plt.show()


plt.figure(figsize=(10, 6))
plt.loglog(x_error_cuda, y_error_cuda, label='CUDA')
plt.loglog(x_error, y_error, label='Séquentiel')
plt.title('Graphique de l\'erreur en échelle log-log rk4')
plt.xlabel('Nombre de Subdivision')
plt.ylabel('Erreur')
plt.legend()
plt.grid(True)
plt.savefig('error.png') 
plt.show()
