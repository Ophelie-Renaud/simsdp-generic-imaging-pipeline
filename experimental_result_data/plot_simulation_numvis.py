import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Liste des fichiers CSV (simulation et mesure)
simulated_sota_csv_files = {
    "g2g_clean": "moldable/simu_sota/g2g_clean.csv",
    "g2g": "moldable/simu_sota/g2g.csv",
    "dft": "moldable/simu_sota/dft.csv",
    "fft": "moldable/simu_sota/fft.csv",
}
simulated_csv_files = {
    "g2g_clean": "moldable/simu/g2g_clean.csv",
    "g2g": "moldable/simu/g2g.csv",
    "dft": "moldable/simu/dft.csv",
    "fft": "moldable/simu/fft.csv",
}

measured_csv_files = {
    "g2g_clean": "moldable/measure/g2g_clean.csv",
    "g2g": "moldable/measure/g2g.csv",
    "dft": "moldable/measure/dft.csv",
    "fft": "moldable/measure/fft.csv",
}

instrumented = {
    "g2g_clean": 90154,
    "g2g": 90154,
    "dft": 97517, # valid
    "fft": 60122, # valid
}

# Ajouter les valeurs extrapolées
extrapolated_data = {
  #  "g2g_clean": {
  #      "sota": [86460., 100590., 114690., 143362.5, 172035.2],
  #      "simu": [44285., 54285., 64285., 75356.25, 86427.5],
  #      "measure": [53005.946, 69645.468, 89454.144, 97486.5, 116983.8]
  #  },
    "g2g": {
        "sota": [65840., 79970., 94070., 117587.5, 141105],
        "simu": [43840., 43840., 43840., 54800, 65760],
        "measure": [43005.946, 49645.468, 49454.144, 79959.5, 95951.4]
    },
    "dft": {
        "sota": [95148., 98548., 101963., 127453.75, 152944.5],
        "simu": [75848., 73793., 71733., 89666.25, 107599.5],
        "measure": [73767.191, 73845.297, 73718.304, 98139.39, 117767.27]
    },
    "fft": {
        "sota": [57623., 61553., 65493., 81866.25, 98239.5],
        "simu": [42083., 42343., 42603., 63253.75, 73904.5],
        "measure": [35700.461, 35607.163, 35769.942, 64674.34, 77609.21]
    }
}

# Fixer les paramètres
fixed_minor_cycle = 150
fixed_grid_size = 1536
#num_vis = np.array([1308160, 1962240, 2616320])
num_vis = np.array([1308160, 1962240, 2616320, 3270400, 3924480])

# Charger le fichier CSV
def load_simu_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, sep=';')
    grid_sizes = sorted(df['GRID_SIZE'].unique())
    num_cycles = sorted(df['NUM_MINOR_CYCLES'].unique())
    num_visibilities = sorted(df['NUM_VISIBILITIES'].unique())
    latency = np.zeros(len(num_visibilities))
    for i, vis in enumerate(num_visibilities):
        latency[i] = df[(df['GRID_SIZE'] == grid_sizes[2]) & 
                        (df['NUM_MINOR_CYCLES'] == num_cycles[2]) & 
                        (df['NUM_VISIBILITIES'] == vis)]['DurationII'].values[0]
    return latency

def load_mes_csv_to_numpy(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.sort_values(by=["NUM_VISIBILITIES", "NUM_MINOR_CYCLES", "GRID_SIZE"])
    num_visibilities = df["NUM_VISIBILITIES"].unique()
    num_cycles = df["NUM_MINOR_CYCLES"].unique()
    grid_sizes = df["GRID_SIZE"].unique()
    latency = np.zeros(len(num_visibilities))
    for i, vis in enumerate(num_visibilities):
        latency[i] = df[(df['GRID_SIZE'] == grid_sizes[2]) & 
                        (df['NUM_MINOR_CYCLES'] == num_cycles[2]) & 
                        (df['NUM_VISIBILITIES'] == vis)]['Latency'].values[0]
    return latency

# Création du plot
plt.figure(figsize=(8, 3))  

# Marqueurs et couleurs modernes pour chaque type de données
markers = {'sota': 'o', 'simu': 's', 'measure': 'd'}
colors = {'g2g_clean': '#1F77B4', 'g2g': '#FF7F0E', 'dft': '#2CA02C', 'fft': '#D62728'}  # Couleurs attrayantes et modernes

    
# Tracer les résultats extrapolés
for key in extrapolated_data.keys():
    simu_sota_data = [value / 1000 for value in extrapolated_data[key]["sota"]]
    simu_data = [value / 1000 for value in extrapolated_data[key]["simu"]]
    measure_data = [value / 1000 for value in extrapolated_data[key]["measure"]]
    
    # Tracer les résultats simulés
    plt.plot(num_vis, simu_data, label=f"{key} - Our model", marker=markers['simu'], linestyle='--', color=colors[key], markersize=8)
    
    # Tracer les résultats mesurés
    plt.plot(num_vis, measure_data, label=f"{key} - Measured", marker=markers['measure'], linestyle=':', color=colors[key], markersize=8)

# Configurer le plot avec des tailles de texte plus grandes pour une meilleure lisibilité
#plt.title(f"Latency vs. Number of Visibilities\n(NUM_MINOR_CYCLES = {fixed_minor_cycle}, GRID_SIZE = {fixed_grid_size})", fontsize=18)
plt.xlabel("Number of Visibilities", fontsize=16)
plt.ylabel("Latency (s)", fontsize=16)
#plt.legend(fontsize=22)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.4)

# Supprimer les bords de la figure
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Augmenter la taille des chiffres sur les axes
plt.tick_params(axis='both', which='major', labelsize=12)

# Déplacer la légende à l'extérieur du graphique
plt.legend(fontsize=14, loc='lower center', bbox_to_anchor=(0.5, -1), ncol=len(extrapolated_data.keys()))
# Ajuster l’espace pour la légende
plt.subplots_adjust(bottom=0.45)

# Exporter le plot en PDF
#plt.tight_layout()  # Ajuster automatiquement les éléments pour éviter la coupure
plt.savefig('latency_vs_visibilities.pdf', format='pdf')

# Afficher le plot
plt.show(block=False)
plt.pause(3)  # Affiche pendant 3 secondes
plt.close()

