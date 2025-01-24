import subprocess
import numpy as np
import os
import re

def compute_average_from_csv(filename):
    result = np.genfromtxt(filename, delimiter=",")
    av = np.mean(result)
    return av

# Exécutable compilé
executable = "./SEP_Pipeline"

# definition de la plage des parametre G = Grid_size, V = nombre de visibilité, C = nombre de cycle mineur, indice 0 = val min, f = val max et s = pas
G0 = 512
Gf = 2560
Gs = 512
V0 = 1000000
Vf = 4000000
Vs = 1000000
C0 = 50
Cf = 250
Cs = 50

# Liste des tests avec plages de valeurs pour certains paramètres
tests = [
    ("time_constant_setups", [3]), #int NUM_SAMPLES
    ("time_gridsize_setups", [3, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_visibility_setups", [3, (V0, Vf, Vs)]), #int NUM_SAMPLES, int NUM_VISIBILITIES
    ("time_save_output", [3, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_dft", [3, (C0, Cf, Cs), (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_MINOR_CYCLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("time_gains_application", [3, (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("time_add_visibilities", [3, (V0, Vf, Vs)]), #int NUM_SAMPLES, int NUM_VISIBILITIES
    ("time_prolate", [3, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_finegrid", [3, (V0, Vf, Vs), V0]), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("time_subtract_ispace", [1, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_fftshift", [3, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_fft", [3, (G0, Gf, Gs)]), #int NUM_SAMPLES, int GRID_SIZE
    ("time_hogbom", [3, (G0, Gf, Gs),(C0, Cf, Cs)]), # int NUM_SAMPLES, int GRID_SIZE, int NUM_MINOR_CYCLES
]

# Fonction pour générer toutes les combinaisons de paramètres
def generate_combinations(params):
    expanded_params = []
    for param in params:
        if isinstance(param, tuple):  # Si c'est une plage (début, fin, pas)
            expanded_params.append(range(param[0], param[1] + 1, param[2]))
        else:  # Sinon, c'est une valeur fixe
            expanded_params.append([param])
    # Génère toutes les combinaisons possibles
    return [list(comb) for comb in np.array(np.meshgrid(*expanded_params)).T.reshape(-1, len(params))]

# Executing actor based on the set up configuration
for test in tests:
    function_name, params = test
    all_combinations = generate_combinations(params)

    for combination in all_combinations:
        command = [executable, function_name] + list(map(str, combination))
        print(f"Lancement : {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Erreur :\n{result.stderr}")


# Répertoire d'entrée et de sortie
input_folder = "to_average"
output_folder = "average"

# Créer le dossier 'average' s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Parcourir les fichiers dans le dossier 'to_average'
for filename in os.listdir(input_folder):
    # Construire le chemin complet du fichier
    full_path = os.path.join(input_folder, filename)

    # Vérifier si c'est un fichier
    if os.path.isfile(full_path):
        # Extraire le nom de base et les paramètres du fichier
        match = re.match(r"(?P<name>\w+)_(?P<d1>\d+)\*(?P<d2>\d+)\*(?P<d3>\d+)", filename)
        if match:
            name = match.group("name")
            d1 = int(match.group("d1"))
            d2 = int(match.group("d2"))
            d3 = int(match.group("d3"))

            # Calculer la moyenne
            average = compute_average_from_csv(full_path)
            if average is not None:
                # Construire le chemin du fichier de sortie
                output_file = os.path.join(output_folder, f"{name}.csv")

                # Ajouter les résultats au fichier de sortie
                with open(output_file, "a") as f:
                    f.write(f"{average}, {d1}, {d2}, {d3},")

                print(f"Résultats ajoutés pour {filename} : {average}")
            else:
                print(f"Impossible de calculer la moyenne pour {filename}.")
        else:
            print(f"Nom de fichier non conforme : {filename}")



