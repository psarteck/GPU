import matplotlib.pyplot as plt

def plot_results(file_name, title):
    # Dictionnaire pour mapper le nombre de threads par bloc à une couleur
    colors = {128: 'red', 256: 'green', 512: 'purple', 1024: 'orange'}

    # Initialisation des listes pour chaque ensemble de données
    data_128 = {'x': [], 'y': []}
    data_256 = {'x': [], 'y': []}
    data_512 = {'x': [], 'y': []}
    data_1024 = {'x': [], 'y': []}

    # Lecture du fichier et remplissage des données
    with open(file_name, 'r') as file:
        current_threads = 0
        for line in file:
            if line.startswith('Threads per block:'):
                current_threads = int(line.split()[-1])
            else:
                x, y = map(float, line.split())
                if current_threads == 128:
                    data_128['x'].append(x)
                    data_128['y'].append(y)
                elif current_threads == 256:
                    data_256['x'].append(x)
                    data_256['y'].append(y)
                elif current_threads == 512:
                    data_512['x'].append(x)
                    data_512['y'].append(y)
                elif current_threads == 1024:
                    data_1024['x'].append(x)
                    data_1024['y'].append(y)

    # Tracer les graphiques
    plt.figure(figsize=(10, 6))

    plt.loglog(data_128['x'], data_128['y'], label='128', color=colors[128])
    plt.loglog(data_256['x'], data_256['y'], label='256', color=colors[256])
    plt.loglog(data_512['x'], data_512['y'], label='512', color=colors[512])
    plt.loglog(data_1024['x'], data_1024['y'], label='1024', color=colors[1024])

    plt.xlabel('Nombre')
    plt.ylabel('Erreur' if 'error' in file_name else 'Temps')
    plt.title(title)
    plt.legend()
    plt.show()

# Exemple d'utilisation
plot_results('error_results_gauss.txt', 'Graphique d\'erreur en fonction du nombre avec différentes tailles de blocs')
plot_results('time_results_gauss.txt', 'Graphique de temps en fonction du nombre avec différentes tailles de blocs')
