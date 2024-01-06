import os

def extract_number_from_filename(filename):
    # Recherche tous les groupes de chiffres dans le nom de fichier
    digits = ''.join(filter(str.isdigit, filename))

    # Convertit les chiffres en entier
    if digits:
        return int(digits)
    else:
        return None

def find_files(directory, prefix):
    # Liste tous les fichiers dans le répertoire
    all_files = os.listdir(directory)

    # Filtrer les fichiers qui commencent par le préfixe
    matching_files = [file for file in all_files if file.startswith(prefix)]

    # Extraire les nombres de chaque fichier
    # file_numbers = {file: extract_number_from_filename(file) for file in matching_files}
    file_numbers = {os.path.join(directory, file): extract_number_from_filename(file) for file in matching_files}

    return file_numbers

