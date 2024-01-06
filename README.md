# CPU / GPU

## Exécution

**Entrez dans le fichier `parametres_Machine` le nombre de processeur max que vous souhaitez utiliser.**

Des scripts bash sont disponibles dans le dossier `Bashs`, ils permettent de lancer les codes pour chaque méthode et chaque paralélisation : 
* Simpson : 
* Gauss2D :
* Runge Kutta 
* MonteCarlo 

Pour les codes en CUDA, ils se trouvent dans le dossier Cuda

## Graph Python 

Le fichier `graph.py` dans le sous répertoire `CodesPython` permet d'afficher les données dans le répertoire `Results`. 
Il prend en entrée le nom préfix des données que vous voulez afficher.
Par exemple pour afficher les données pour la méthode de Simpson avec Open MP, placez vous dans le répertoire `CodesPython` et tappez : 
```
python3 graph.py "simp_Op_MP"
```