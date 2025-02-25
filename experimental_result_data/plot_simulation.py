import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
    x_simu_sota, y_simu_sota, z_simu_sota, c_simu_sota = df_simu_sota["GRID_SIZE"], df_simu_sota["NUM_VISIBILITIES"], df_simu_sota[subset_name], df_simu_sota["NUM_MINOR_CYCLES"]
    x_simu, y_simu, z_simu, c_simu = df_simu["GRID_SIZE"], df_simu["NUM_VISIBILITIES"], df_simu[subset_name], df_simu["NUM_MINOR_CYCLES"]
    x_meas, y_meas, z_meas, c_meas = df_measure["GRID_SIZE"], df_measure["NUM_VISIBILITIES"], df_measure[subset_name], df_measure["NUM_MINOR_CYCLES"]

    #z_simu_sota = np.array(z_simu_sota) * 1000  # Conversion sec → ms

    x_min, x_max = min(x_simu_sota.min(), x_simu.min(), x_meas.min()), max(x_simu_sota.max(), x_simu.max(), x_meas.max())
    y_min, y_max = min(y_simu_sota.min(), y_simu.min(), y_meas.min()), max(y_simu_sota.max(), y_simu.max(), y_meas.max())
    z_min, z_max = min(z_simu_sota.min(), z_simu.min(), z_meas.min()), max(z_simu_sota.max(), z_simu.max(), z_meas.max())
    c_min, c_max = min(c_simu_sota.min(), c_simu.min(), c_meas.min()), max(c_simu_sota.max(), c_simu.max(), c_meas.max())
    
    print(f"x_min: {x_min}, x_max: {x_max}")
    print(f"y_min: {y_min}, y_max: {y_max}")
    print(f"z_min: {z_min}, z_max: {z_max}")
    print(f"c_min: {c_min}, c_max: {c_max}")

    # Vérification des dimensions
    if len(z_simu_sota) > len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> Ajustement en cours")
        z_simu_sota = z_simu_sota[:len(z_meas)]  # Tronquer
        y_simu_sota = y_simu_sota[:len(y_meas)]
        x_simu_sota = x_simu_sota[:len(x_meas)]
        c_simu_sota = c_simu_sota[:len(c_meas)]
    elif len(z_simu_sota) < len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> Ajustement en cours")
        z_meas = z_meas[:len(z_simu_sota)]  # Tronquer
        y_meas = y_meas[:len(y_simu_sota)]
        x_meas = x_meas[:len(x_simu_sota)]
        c_meas = c_meas[:len(c_simu_sota)]
            
    rmse_sota = compute_rmse(z_simu_sota, z_meas)
    error_sota = (rmse_sota/ np.mean(z_meas))*100
    
    if len(z_simu) > len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> Ajustement en cours")
        z_simu = z_simu[:len(z_meas)]  # Tronquer
        y_simu = y_simu[:len(y_meas)]
        x_simu = x_simu[:len(x_meas)]
        c_simu = c_simu[:len(c_meas)]
        rmse = compute_rmse(z_simu, z_meas)
    elif len(z_simu) > len(z_meas):
        print(f"Attention : Dimensions différentes pour {file_key} -> Ajustement en cours")
        z_meas = z_meas[:len(z_simu)]  # Tronquer
        y_meas = y_meas[:len(y_simu)]
        x_meas = x_meas[:len(x_simu)]
        c_meas = c_meas[:len(c_simu)]
        
    rmse = compute_rmse(z_simu, z_meas)
    error = (rmse/ np.mean(z_meas))*100

    # Création des grilles pour interpolation
    xi = np.linspace(min(x_simu), max(x_simu), 50)
    yi = np.linspace(min(y_simu), max(y_simu), 50)
    X, Y = np.meshgrid(xi, yi)

    Z_simu_sota = griddata((x_simu_sota, y_simu_sota), z_simu_sota, (X, Y), method='cubic')
    Z_simu = griddata((x_simu, y_simu), z_simu, (X, Y), method='cubic')
    Z_meas = griddata((x_meas, y_meas), z_meas, (X, Y), method='cubic')
    
    c_value = [50, 100, 150] #, 200, 250]
    colors = ['#3498db', '#2ecc71', '#f1c40f'] #, '#e67e22', '#e74c3c']
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([45, 75, 125, 175], cmap.N)
    #norm = mcolors.BoundaryNorm([45, 75, 125, 175, 225, 275], cmap.N)


    # Création de la figure avec 2 sous-graphes
    fig, axes = plt.subplots(1, 3, figsize=(22, 5), subplot_kw={'projection': '3d'})
    
        # Simulation
    surf0 = axes[0].plot_surface(X, Y, Z_simu_sota, cmap=cmap, edgecolor='none', alpha=0.9)
    title = f"Simulation S. Wang et. al. - {file_key}"
    if rmse is not None:
        title += f"\nRMSE = {rmse_sota:.2f}\nError = {error_sota:.2f}%"
    axes[0].set_title(title)
    axes[0].set_xlabel("GRID_SIZE")
    axes[0].set_ylabel("NUM_VISIBILITIES")
    axes[0].set_zlabel(subset_name)
    
    # Ajustement des limites des axes
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_zlim(z_min, z_max)
    
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0], shrink=0.5, aspect=15, ticks=c_value)
    cbar.set_label("NUM_MINOR_CYCLES")
    
    # Simulation
    surf1 = axes[1].plot_surface(X, Y, Z_simu, cmap=cmap, edgecolor='none', alpha=0.9)
    title = f"Simulation - {file_key}"
    if rmse is not None:
        title += f"\nRMSE = {rmse:.2f}\nError = {error:.2f}%"
    axes[1].set_title(title)
    axes[1].set_xlabel("GRID_SIZE")
    axes[1].set_ylabel("NUM_VISIBILITIES")
    axes[1].set_zlabel(subset_name)
    
    # Ajustement des limites des axes
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_zlim(z_min, z_max)
    
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[1], shrink=0.5, aspect=15, ticks=c_value)
    cbar.set_label("NUM_MINOR_CYCLES")

    # Mesure
    surf2 = axes[2].plot_surface(X, Y, Z_meas, cmap=cmap, edgecolor='none', alpha=0.9)
    title = f"Mesure - {file_key}"
    axes[2].set_title(title)
    axes[2].set_xlabel("GRID_SIZE")
    axes[2].set_ylabel("NUM_VISIBILITIES")
    axes[2].set_zlabel(subset_name)
    
    # Ajustement des limites des axes
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)
    axes[2].set_zlim(z_min, z_max)
    
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[2], shrink=0.5, aspect=15, ticks=c_value)
    cbar.set_label("NUM_MINOR_CYCLES")

    plt.tight_layout()

    # Sauvegarde de la figure
    fig.savefig(f"3D_comparison_{file_key}.png", format="png", dpi=400)
    print(f"Graphique enregistré : 3D_comparison_{file_key}.png")

    plt.show(block=False)

# Boucle sur chaque fichier pour comparer simulations et mesures
for key in simulated_csv_files.keys():
    plot_3d_comparison(key, simulated_sota_csv_files[key], simulated_csv_files[key], measured_csv_files[key])

