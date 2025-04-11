import math
import subprocess
import numpy as np
import os
import re
import shutil
from collections import defaultdict

"""
Script de lancement des calculs composant les pipeline d'imagerie de radio astronomie.

Ce script exécute une série de calculs depuis l'exécutable SEP_Pipeline
en variant plusieurs paramètres (taille de grille, visibilité, etc.).
Les résultats sont collectés et moyennés pour une analyse plus précise.

Fonctionnalités :
- Suppression des fichiers temporaires avant exécution.
- Génération de combinaisons de paramètres pour tester différents scénarios.
- Exécution des tests et collecte des résultats.
- Calcul des moyennes à partir des fichiers de sortie CSV.
- Stockage des résultats moyennés dans un dossier spécifique.

Auteur : Ophélie RENAUD
Date : 7/02/2025
"""

# ================================
# Configuration et Paramètres
# ================================
INPUT_FOLDER = "to_average"
OUTPUT_FOLDER = "average"
EXECUTABLE = "./SEP_Pipeline"
NUM_SAMPLES = 10
#MEM_MAX = 1 * 1024*1024*1024 #1Go
#MEM_MAX = 500 * 1024*1024 #500Mo
MEM_MAX = 1 * 1024*1024 #1Mo

# Plages de paramètres pour les executions
#G = Grid_size, V = nombre de visibilité, C = nombre de cycle mineur, indice 0 = val min, f = val max et s = pas
Gt0 = 100
Gtf = int(math.sqrt(MEM_MAX / 4)) #float limitation ->512
Gts = int((Gtf - Gt0) / 5) #->82
G0,Gf,Gs=512,2560,512 #5
V0, Vf, Vs = 3924480, 19622400, 3924480 #5
C0, Cf, Cs = 50, 400, 50 #8


# ================================
# Fonctions Utilitaires
# ================================
def clear_folder(folder_path):
    """Supprime tout le contenu d'un dossier donné."""
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Supprime les fichiers et liens symboliques
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Supprime les sous-dossiers et leur contenu

def compute_average_from_csv(filename):
    """Calcule la moyenne des N echantillons par configurations contenues dans un fichier CSV."""
    result = np.genfromtxt(filename, delimiter=",")
    av = np.mean(result)
    return av

def generate_combinations(params):
    """Génère toutes les combinaisons possibles des paramètres donnés."""
    expanded_params = []
    for param in params:
        if isinstance(param, tuple):  # Si c'est une plage (début, fin, pas)
            expanded_params.append(range(param[0], param[1] + 1, param[2]))
        else:  # Sinon, c'est une valeur fixe
            expanded_params.append([param])
    # Génère toutes les combinaisons possibles
    meshgrid_result = np.meshgrid(*expanded_params)

    # Réorganiser le résultat pour obtenir les bonnes combinaisons
    combinations = [list(comb) for comb in zip(*[x.flatten() for x in meshgrid_result])]

    return combinations

def extract_parts(filename):
    """Sépare le préfixe et les nombres"""
    match = re.match(r"(.+?)_(\d+)\*(\d+)\*(\d+)", filename)
    if match:
        prefix = match.group(1)  # Partie avant les nombres
        numbers = tuple(map(int, match.groups()[1:]))  # Convertir les nombres en entiers
        return prefix, numbers
    return filename, (float('inf'), float('inf'), float('inf'))  # Mettre en dernier les fichiers non conformes


# ================================
# Définition des Tests
# ================================
tests = [

        ("time_psf_host_set_up", [NUM_SAMPLES, (G0, Gf, Gs)]),
         ]

#test_unit = [
#        ("time_constant_setups", [NUM_SAMPLES]), #int NUM_SAMPLES
#     ("time_gridsize_setups", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#    ("time_visibility_setups", [NUM_SAMPLES, (V0, Vf, Vs)]), #int NUM_SAMPLES, int NUM_VISIBILITIES
#     ("time_save_output", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#     ("time_dft", [NUM_SAMPLES, (C0, Cf, Cs), (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_MINOR_CYCLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
#     ("time_gains_application", [NUM_SAMPLES, (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
#     ("time_substraction", [NUM_SAMPLES, (V0, Vf, Vs)]), #int NUM_SAMPLES, int NUM_VISIBILITIES
#     ("time_add_visibilities", [NUM_SAMPLES, (V0, Vf, Vs)]), #int NUM_SAMPLES, int NUM_VISIBILITIES
#     ("time_prolate", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#     ("time_finegrid", [NUM_SAMPLES, (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
#    ("time_subtract_ispace", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#     ("time_fft_shift", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#     ("time_fft", [NUM_SAMPLES, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
#     ("time_hogbom", [NUM_SAMPLES, (G0, Gf, Gs),(C0, Cf, Cs)]), # int NUM_SAMPLES, int GRID_SIZE, int NUM_MINOR_CYCLES
#     ("time_std_gridding", [NUM_SAMPLES, (G0, Gf, Gs),(V0, Vf, Vs)]),
#     ("time_dft_gridding", [NUM_SAMPLES, (G0, Gf, Gs),(V0, Vf, Vs)]),
#     ("time_degrid", [NUM_SAMPLES, (G0, Gf, Gs),(V0, Vf, Vs)]),
#     ("time_s2s", [NUM_SAMPLES, (G0, Gf, Gs),(V0, Vf, Vs)]),
#     ("time_psf_host_set_up", [NUM_SAMPLES, (G0, Gf, Gs)]),
#]

# ================================
# Exécution des Tests
# ================================

clear_folder(INPUT_FOLDER)
clear_folder(OUTPUT_FOLDER)

#for test in test_unit:
for test in tests:
    function_name, params = test
    all_combinations = generate_combinations(params)

    for combination in all_combinations:
        command = [EXECUTABLE, function_name] + list(map(str, combination))
        print(f"Lancement : {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Erreur :\n{result.stderr}")



# ================================
# Traitement des Résultats
# ================================
# Créer le dossier 'average' s'il n'existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Regrouper les fichiers par préfixe
files_by_prefix = defaultdict(list)

for filename in os.listdir(INPUT_FOLDER):
    prefix, numbers = extract_parts(filename)
    files_by_prefix[prefix].append((filename, numbers))

# Trier les fichiers dans chaque groupe
sorted_files = []
for prefix in sorted(files_by_prefix.keys()):  # Trier les préfixes par ordre alphabétique
    sorted_files.extend(sorted(files_by_prefix[prefix], key=lambda x: x[1]))  # Trier par les nombres

# Parcourir les fichiers triés
for filename in sorted_files:
    # Construire le chemin complet du fichier
    full_path = os.path.join(INPUT_FOLDER, filename[0])
    print(full_path)

    # Vérifier si c'est un fichier
    if os.path.isfile(full_path):
        # Extraire le nom de base et les paramètres du fichier
        match = re.match(r"(?P<name>\w+)_(?P<d1>\d+)\*(?P<d2>\d+)\*(?P<d3>\d+)", filename[0])
        if match:
            name = match.group("name")
            d1 = int(match.group("d1"))
            d2 = int(match.group("d2"))
            d3 = int(match.group("d3"))

            # Calculer la moyenne
            average = compute_average_from_csv(full_path)
            if average is not None:
                # Construire le chemin du fichier de sortie
                output_file = os.path.join(OUTPUT_FOLDER, f"{name}.csv")

                # Ajouter les résultats au fichier de sortie
                with open(output_file, "a") as f:
                    if d3!=0:
                        f.write(f"{average}, {d1}, {d2}, {d3},")
                    elif d2!=0:
                        f.write(f"{average}, {d1}, {d2},")
                    else:
                        f.write(f"{average}, {d1},")

                print(f"Résultats ajoutés pour {filename} : {average}")
            else:
                print(f"Impossible de calculer la moyenne pour {filename}.")
        else:
            print(f"Nom de fichier non conforme : {filename}")



