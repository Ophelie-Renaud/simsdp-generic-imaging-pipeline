import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Liste des fichiers CSV à traiter
csv_files = {
    "g2g_clean": "moldable/g2g_clean.csv",
    "g2g": "moldable/g2g.csv",
    #"dft": "moldable/dft.csv",
    #"fft": "moldable/fft.csv"
}

subset_name = "DurationII"  # Ou "Latency"

def plot_data(file_key, file_path):
    # Chargement des données
    df = pd.read_csv(file_path, delimiter=';')
    
    # Vérifier que les colonnes existent
    required_columns = {"NUM_MINOR_CYCLES", "NUM_VISIBILITIES", "GRID_SIZE", subset_name}
    if not required_columns.issubset(df.columns):
        print(f"Erreur : Colonnes manquantes dans {file_path}")
        return
    
    # Création d'une figure avec trois sous-graphiques
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
    markers = ['o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '*']
    
    # --- 1. Latency vs GRID_SIZE ---
    for (num_cycles, num_vis) in itertools.product(df["NUM_MINOR_CYCLES"].unique(), df["NUM_VISIBILITIES"].unique()):
        subset = df[(df["NUM_MINOR_CYCLES"] == num_cycles) & (df["NUM_VISIBILITIES"] == num_vis)]
        axes[0].plot(subset["GRID_SIZE"], subset[subset_name], 
                     marker=markers[(int(num_cycles) // 50) % len(markers)], 
                     linestyle='-', color=colors[(int(num_vis) // 1308160) % len(colors)], 
                     label=f"C={num_cycles}, V={num_vis}")
    
    axes[0].set_xlabel("GRID_SIZE")
    axes[0].set_ylabel("Latency")
    axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
    axes[0].grid()
    
    # --- 2. Latency vs NUM_MINOR_CYCLES ---
    for (grid_size, num_vis) in itertools.product(df["GRID_SIZE"].unique(), df["NUM_VISIBILITIES"].unique()):
        subset = df[(df["GRID_SIZE"] == grid_size) & (df["NUM_VISIBILITIES"] == num_vis)]
        axes[1].plot(subset["NUM_MINOR_CYCLES"], subset[subset_name], 
                     marker=markers[(int(grid_size) // 512) % len(markers)], 
                     linestyle='-', color=colors[(int(num_vis) // 1308160) % len(colors)], 
                     label=f"G={grid_size}, V={num_vis}")
    
    axes[1].set_xlabel("NUM_MINOR_CYCLES")
    axes[1].set_ylabel("Latency")
    axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
    axes[1].grid()
    
    # --- 3. Latency vs NUM_VISIBILITIES ---
    for (grid_size, num_cycles) in itertools.product(df["GRID_SIZE"].unique(), df["NUM_MINOR_CYCLES"].unique()):
        subset = df[(df["GRID_SIZE"] == grid_size) & (df["NUM_MINOR_CYCLES"] == num_cycles)]
        axes[2].plot(subset["NUM_VISIBILITIES"], subset[subset_name], 
                     marker=markers[(int(grid_size) // 512) % len(markers)], 
                     linestyle='-', color=colors[(int(num_cycles) // 50) % len(colors)], 
                     label=f"G={grid_size}, C={num_cycles}")
    
    axes[2].set_xlabel("NUM_VISIBILITIES")
    axes[2].set_ylabel("Latency")
    axes[2].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
    axes[2].grid()
    
    # Titre global et ajustement de l'affichage
    fig.suptitle(f"{file_key} Latency vs GRID_SIZE, NUM_MINOR_CYCLES, and NUM_VISIBILITIES", fontsize=16)
    plt.tight_layout()
    
    # Enregistrer la figure en PNG
    fig.savefig(f"simulation_{file_key}.png", format="png", dpi=400)
    print(f"Graphique enregistré : simulation_{file_key}.png")
    
    plt.show()
    
def plot_3d_data(file_key, file_path):
    # Chargement des données
    df = pd.read_csv(file_path, delimiter=';')

    # Vérifier que les colonnes existent
    required_columns = {"NUM_MINOR_CYCLES", "NUM_VISIBILITIES", "GRID_SIZE", subset_name}
    if not required_columns.issubset(df.columns):
        print(f"Erreur : Colonnes manquantes dans {file_path}")
        return

    # Création d'une figure 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Conversion en tableau numpy
    x = df["GRID_SIZE"].to_numpy()
    y = df["NUM_MINOR_CYCLES"].to_numpy()
    z = df["NUM_VISIBILITIES"].to_numpy()
    c = df[subset_name].to_numpy()  # Valeurs de DurationII ou Latency

    # Création du scatter plot 3D
    sc = ax.scatter(x, y, z, c=c, cmap='viridis', marker='o', s=50, edgecolors='k', alpha=0.8)

    # Ajout d'une barre de couleur
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(subset_name)

    # Labels et titre
    ax.set_xlabel("GRID_SIZE")
    ax.set_ylabel("NUM_MINOR_CYCLES")
    ax.set_zlabel("NUM_VISIBILITIES")
    ax.set_title(f"3D Scatter Plot - {file_key}")

    # Ajustement de l'affichage
    plt.tight_layout()

    # Sauvegarde du graphique
    fig.savefig(f"3D_plot_{file_key}.png", format="png", dpi=400)
    print(f"Graphique 3D enregistré : 3D_plot_{file_key}.png")

    plt.show()   
    
def plot_3d_surface(file_key, file_path):
    # Chargement des données
    df = pd.read_csv(file_path, delimiter=';')

    # Vérifier que les colonnes existent
    required_columns = {"NUM_MINOR_CYCLES", "NUM_VISIBILITIES", "GRID_SIZE", subset_name}
    if not required_columns.issubset(df.columns):
        print(f"Erreur : Colonnes manquantes dans {file_path}")
        return

    # Extraire les colonnes
    x = df["GRID_SIZE"].to_numpy()
    y = df["NUM_MINOR_CYCLES"].to_numpy()
    z = df[subset_name].to_numpy()  # Latency ou DurationII

    # Création d'une grille régulière pour l'interpolation
    xi = np.linspace(min(x), max(x), 50)
    yi = np.linspace(min(y), max(y), 50)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')  # Interpolation cubique

    # Création de la figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Tracer la surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Ajouter une barre de couleur
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label(subset_name)

    # Labels et titre
    ax.set_xlabel("GRID_SIZE")
    ax.set_ylabel("NUM_MINOR_CYCLES")
    ax.set_zlabel(subset_name)
    ax.set_title(f"Surface 3D - {file_key}")

    # Sauvegarde de la figure
    fig.savefig(f"3D_surface_{file_key}.png", format="png", dpi=400)
    print(f"Graphique 3D enregistré : 3D_surface_{file_key}.png")

    plt.show()


# Exécuter la fonction pour chaque fichier
for key, path in csv_files.items():
    #plot_data(key, path)
    #plot_3d_data(key, path)
    plot_3d_surface(key, path)

