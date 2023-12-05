import numpy as np
import matplotlib.pyplot as plt
import sys 

if len(sys.argv) > 1:
    algo = sys.argv[1]
else:
    algo = "data"

data2 = np.loadtxt("Results/"+algo+"_nbProc_2.txt")
data3 = np.loadtxt("Results/"+algo+"_nbProc_3.txt")
data4 = np.loadtxt("Results/"+algo+"_nbProc_4.txt")
data5 = np.loadtxt("Results/"+algo+"_nbProc_5.txt")
data6 = np.loadtxt("Results/"+algo+"_nbProc_6.txt")

plt.figure(figsize=(10,8))

plt.plot(data2[:,0], data2[:,1], linewidth=4, label="2 processeur")
# plt.plot(data3[:,0], data3[:,1], linewidth=4, label="3 processeur")
plt.plot(data4[:,0], data4[:,1], linewidth=4, label="4 processeur")
plt.plot(data5[:,0], data5[:,1], linewidth=4, label="5 processeur")
plt.plot(data6[:,0], data6[:,1], linewidth=4, label="6 processeur")
plt.title("Etude de l'erreur en fonction du nombre de subdivision",fontsize=18, fontweight='bold')
plt.xlabel("Nombre de subdivision n",fontsize=14)
plt.ylabel("Erreur",fontsize=14)
plt.legend()

plt.figure(figsize=(10,8))

plt.plot(data2[:,0], data2[:,2], linewidth=4, label="2 processeur")
# plt.plot(data3[:,0], data3[:,2], linewidth=4, label="3 processeur")
plt.plot(data4[:,0], data4[:,2], linewidth=4, label="4 processeur")
plt.plot(data5[:,0], data5[:,2], linewidth=4, label="5 processeur")
plt.plot(data6[:,0], data6[:,2], linewidth=4, label="6 processeur")
plt.title("Etude du temps en fonction du nombre de subdivision",fontsize=18, fontweight='bold')
plt.xlabel("Nombre de subdivision n",fontsize=14)
plt.ylabel("Temps [s]",fontsize=14)
# plt.xscale("log")
# plt.yscale("log")
plt.legend()

plt.show()