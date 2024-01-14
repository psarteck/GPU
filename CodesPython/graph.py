import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from utiles_functions import find_files
from utiles_functions import extract_number_from_filename

if len(sys.argv) > 1:
    algo = sys.argv[1]
else:
    algo = "data"

directory_path = "../Results"

files = find_files(directory_path, algo)


# Affichage des données
plt.figure(figsize=(10,8))
for file, number in files.items():
    datas = np.loadtxt(file)
    plt.plot(datas[:,0], datas[:,1], '-o', linewidth=4, label=str(number)+" processeurs")
plt.title("Etude de l'erreur en fonction du nombre de subdivision",fontsize=18, fontweight='bold')
plt.xlabel("Nombre de subdivision n",fontsize=14)
plt.ylabel("Erreur",fontsize=14)
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("../Images/error_"+algo+".png")

plt.figure(figsize=(10,8))
for file, number in files.items():
    datas = np.loadtxt(file)
    plt.plot(datas[:,0], datas[:,2], '-o', linewidth=4, label=str(number)+" processeurs")
plt.title("Etude du temps en fonction du nombre de subdivision",fontsize=18, fontweight='bold')
plt.xlabel("Nombre de subdivision n",fontsize=14)
plt.ylabel("Temps [s]",fontsize=14)
plt.xscale("log", base=2)
plt.yscale("log")
plt.legend()
plt.savefig("../Images/time_"+algo+".png")
plt.show()
