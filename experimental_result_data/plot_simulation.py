import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error

# Liste des fichiers CSV (simulation et mesure)
simulated_sota_csv_files = {
    "g2g_clean": "moldable/simu_sota/g2g_clean.csv",
    "g2g": "moldable/simu_sota/g2g.csv",
}
simulated_csv_files = {
    "g2g_clean": "moldable/simu/g2g_clean.csv",
    "g2g": "moldable/simu/g2g.csv",
}

measured_csv_files = {
    "g2g_clean": "moldable/measure/g2g_clean.csv",
    "g2g": "moldable/measure/g2g.csv",
}

subset_name = "DurationII"  # Colonne d'intérêt (ex: "Latency" ou "DurationII")

def load_data(file_path):
    """ Charge un fichier CSV et extrait les colonnes utiles """
    df = pd.read_csv(file_path, delimiter=';')

    # Vérifier la présence des colonnes nécessaires
    required_columns = {"NUM_MINOR_CYCLES", "NUM_VISIBILITIES", "GRID_SIZE", subset_name}
    if not required_columns.issubset(df.columns):
        print(f"Erreur : Colonnes manquantes dans {file_path}")
        return None

    return df

def compute_rmse(simulated, measured):
    """ Calcule la RMSE entre les valeurs simulées et mesurées """
    return np.sqrt(mean_squared_error(simulated, measured))

def plot_3d_comparison(file_key,simu_sota_path, simu_path, measure_path):
    """ Affiche les surfaces 3D des simulations et mesures côte à côte, et calcule la RMSE """
    
    # Charger les données
    df_simu_sota = load_data(simu_sota_path)
    df_simu = load_data(simu_path)
    df_measure = load_data(measure_path)
    if df_simu is None or df_measure is None:
        return

    # Extraire les colonnes
    x_simu_sota, y_simu_sota, z_simu_sota = df_simu_sota["GRID_SIZE"], df_simu_sota["NUM_VISIBILITIES"], df_simu_sota[subset_name]
    x_simu, y_simu, z_simu = df_simu["GRID_SIZE"], df_simu["NUM_VISIBILITIES"], df_simu[subset_name]
    x_meas, y_meas, z_meas = df_measure["GRID_SIZE"], df_measure["NUM_VISIBILITIES"], df_measure[subset_name]

    # Vérification des dimensions
    if len(z_simu_sota) != len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> RMSE non calculée")
        rmse_sota = None
    else:
        rmse_sota = compute_rmse(z_simu_sota, z_meas)
        
    if len(z_simu) != len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> RMSE non calculée")
        rmse = None
    else:
        rmse = compute_rmse(z_simu, z_meas)

    # Création des grilles pour interpolation
    xi = np.linspace(min(x_simu), max(x_simu), 50)
    yi = np.linspace(min(y_simu), max(y_simu), 50)
    X, Y = np.meshgrid(xi, yi)

    Z_simu_sota = griddata((x_simu_sota, y_simu_sota), z_simu_sota, (X, Y), method='cubic')
    Z_simu = griddata((x_simu, y_simu), z_simu, (X, Y), method='cubic')
    Z_meas = griddata((x_meas, y_meas), z_meas, (X, Y), method='cubic')

    # Création de la figure avec 2 sous-graphes
    fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': '3d'})
    
        # Simulation
    surf0 = axes[0].plot_surface(X, Y, Z_simu_sota, cmap='viridis', edgecolor='none', alpha=0.9)
    title = f"Simulation S. Wang et. al. - {file_key}"
    if rmse is not None:
        title += f"\nRMSE = {rmse_sota:.2f}"
    axes[0].set_title(title)
    axes[0].set_xlabel("GRID_SIZE")
    axes[0].set_ylabel("NUM_VISIBILITIES")
    axes[0].set_zlabel(subset_name)
    fig.colorbar(surf0, ax=axes[0], shrink=0.5, aspect=15)

    # Simulation
    surf1 = axes[1].plot_surface(X, Y, Z_simu, cmap='viridis', edgecolor='none', alpha=0.9)
    title = f"Simulation - {file_key}"
    if rmse is not None:
        title += f"\nRMSE = {rmse:.2f}"
    axes[1].set_title(title)
    axes[1].set_xlabel("GRID_SIZE")
    axes[1].set_ylabel("NUM_VISIBILITIES")
    axes[1].set_zlabel(subset_name)
    fig.colorbar(surf1, ax=axes[1], shrink=0.5, aspect=15)

    # Mesure
    surf2 = axes[2].plot_surface(X, Y, Z_meas, cmap='plasma', edgecolor='none', alpha=0.9)
    title = f"Mesure - {file_key}"
    axes[2].set_title(title)
    axes[2].set_xlabel("GRID_SIZE")
    axes[2].set_ylabel("NUM_VISIBILITIES")
    axes[2].set_zlabel(subset_name)
    fig.colorbar(surf2, ax=axes[2], shrink=0.5, aspect=15)
    
    #plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2, hspace=0.2)


    plt.tight_layout()

    # Sauvegarde de la figure
    fig.savefig(f"3D_comparison_{file_key}.png", format="png", dpi=400)
    print(f"Graphique enregistré : 3D_comparison_{file_key}.png")

    plt.show()

# Boucle sur chaque fichier pour comparer simulations et mesures
for key in simulated_csv_files.keys():
    plot_3d_comparison(key, simulated_sota_csv_files[key], simulated_csv_files[key], measured_csv_files[key])

