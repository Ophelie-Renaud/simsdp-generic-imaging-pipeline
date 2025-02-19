import pandas as pd
import matplotlib.pyplot as plt
import itertools

# Données au format CSV
data = """GRID_SIZE;NUM_MINOR_CYCLES;NUM_VISIBILITIES;Schedulability;Power;Latency;DurationII;Memory;AskedCuts;AskedPrecuts
512.0;50.0;1308160.0;true;211;2;7242;0;0;0
1024.0;50.0;1308160.0;true;211;2;7306;0;0;0
1536.0;50.0;1308160.0;true;211;2;7390;0;0;0
2048.0;50.0;1308160.0;true;211;2;7528;0;0;0
2560.0;50.0;1308160.0;true;211;2;7713;0;0;0
512.0;100.0;1308160.0;true;211;2;7242;0;0;0
1024.0;100.0;1308160.0;true;211;2;7306;0;0;0
1536.0;100.0;1308160.0;true;211;2;7390;0;0;0
2048.0;100.0;1308160.0;true;211;2;7528;0;0;0
2560.0;100.0;1308160.0;true;211;2;7713;0;0;0
512.0;150.0;1308160.0;true;211;2;7242;0;0;0
1024.0;150.0;1308160.0;true;211;2;7306;0;0;0
1536.0;150.0;1308160.0;true;211;2;7390;0;0;0
2048.0;150.0;1308160.0;true;211;2;7528;0;0;0
2560.0;150.0;1308160.0;true;211;2;7713;0;0;0
512.0;200.0;1308160.0;true;211;2;7242;0;0;0
1024.0;200.0;1308160.0;true;211;2;7306;0;0;0
1536.0;200.0;1308160.0;true;211;2;7390;0;0;0
2048.0;200.0;1308160.0;true;211;2;7528;0;0;0
2560.0;200.0;1308160.0;true;211;2;7713;0;0;0
512.0;250.0;1308160.0;true;211;2;7242;0;0;0
1024.0;250.0;1308160.0;true;211;2;7306;0;0;0
1536.0;250.0;1308160.0;true;211;2;7390;0;0;0
2048.0;250.0;1308160.0;true;211;2;7528;0;0;0
2560.0;250.0;1308160.0;true;211;2;7713;0;0;0
512.0;50.0;1962240.0;true;211;2;7242;0;0;0
1024.0;50.0;1962240.0;true;211;2;7306;0;0;0
1536.0;50.0;1962240.0;true;211;2;7390;0;0;0
2048.0;50.0;1962240.0;true;211;2;7528;0;0;0
2560.0;50.0;1962240.0;true;211;2;7713;0;0;0
512.0;100.0;1962240.0;true;211;2;7242;0;0;0
1024.0;100.0;1962240.0;true;211;2;7306;0;0;0
1536.0;100.0;1962240.0;true;211;2;7390;0;0;0
2048.0;100.0;1962240.0;true;211;2;7528;0;0;0
2560.0;100.0;1962240.0;true;211;2;7713;0;0;0
512.0;150.0;1962240.0;true;211;2;7242;0;0;0
1024.0;150.0;1962240.0;true;211;2;7306;0;0;0
1536.0;150.0;1962240.0;true;211;2;7390;0;0;0
2048.0;150.0;1962240.0;true;211;2;7528;0;0;0
2560.0;150.0;1962240.0;true;211;2;7713;0;0;0
512.0;200.0;1962240.0;true;211;2;7242;0;0;0
1024.0;200.0;1962240.0;true;211;2;7306;0;0;0
1536.0;200.0;1962240.0;true;211;2;7390;0;0;0
2048.0;200.0;1962240.0;true;211;2;7528;0;0;0
2560.0;200.0;1962240.0;true;211;2;7713;0;0;0
512.0;250.0;1962240.0;true;211;2;7242;0;0;0
1024.0;250.0;1962240.0;true;211;2;7306;0;0;0
1536.0;250.0;1962240.0;true;211;2;7390;0;0;0
2048.0;250.0;1962240.0;true;211;2;7528;0;0;0
2560.0;250.0;1962240.0;true;211;2;7713;0;0;0
512.0;50.0;2616320.0;true;211;2;7244;0;0;0
1024.0;50.0;2616320.0;true;211;2;7307;0;0;0
1536.0;50.0;2616320.0;true;211;2;7390;0;0;0
2048.0;50.0;2616320.0;true;211;2;7528;0;0;0
2560.0;50.0;2616320.0;true;211;2;7713;0;0;0
512.0;100.0;2616320.0;true;211;2;7244;0;0;0
1024.0;100.0;2616320.0;true;211;2;7307;0;0;0
1536.0;100.0;2616320.0;true;211;2;7390;0;0;0
2048.0;100.0;2616320.0;true;211;2;7528;0;0;0
2560.0;100.0;2616320.0;true;211;2;7713;0;0;0
512.0;150.0;2616320.0;true;211;2;7244;0;0;0
1024.0;150.0;2616320.0;true;211;2;7307;0;0;0
1536.0;150.0;2616320.0;true;211;2;7390;0;0;0
2048.0;150.0;2616320.0;true;211;2;7528;0;0;0
2560.0;150.0;2616320.0;true;211;2;7713;0;0;0
512.0;200.0;2616320.0;true;211;2;7244;0;0;0
1024.0;200.0;2616320.0;true;211;2;7307;0;0;0
1536.0;200.0;2616320.0;true;211;2;7390;0;0;0
2048.0;200.0;2616320.0;true;211;2;7528;0;0;0
2560.0;200.0;2616320.0;true;211;2;7713;0;0;0
512.0;250.0;2616320.0;true;211;2;7244;0;0;0
1024.0;250.0;2616320.0;true;211;2;7307;0;0;0
1536.0;250.0;2616320.0;true;211;2;7390;0;0;0
2048.0;250.0;2616320.0;true;211;2;7528;0;0;0
2560.0;250.0;2616320.0;true;211;2;7713;0;0;0
512.0;50.0;3270400.0;true;211;2;7246;0;0;0
1024.0;50.0;3270400.0;true;211;2;7308;0;0;0
1536.0;50.0;3270400.0;true;211;2;7390;0;0;0
2048.0;50.0;3270400.0;true;211;2;7528;0;0;0
2560.0;50.0;3270400.0;true;211;2;7713;0;0;0
512.0;100.0;3270400.0;true;211;2;7246;0;0;0
1024.0;100.0;3270400.0;true;211;2;7308;0;0;0
1536.0;100.0;3270400.0;true;211;2;7390;0;0;0
2048.0;100.0;3270400.0;true;211;2;7528;0;0;0
2560.0;100.0;3270400.0;true;211;2;7713;0;0;0
512.0;150.0;3270400.0;true;211;2;7246;0;0;0
1024.0;150.0;3270400.0;true;211;2;7308;0;0;0
1536.0;150.0;3270400.0;true;211;2;7390;0;0;0
2048.0;150.0;3270400.0;true;211;2;7528;0;0;0
2560.0;150.0;3270400.0;true;211;2;7713;0;0;0
512.0;200.0;3270400.0;true;211;2;7246;0;0;0
1024.0;200.0;3270400.0;true;211;2;7308;0;0;0
1536.0;200.0;3270400.0;true;211;2;7390;0;0;0
2048.0;200.0;3270400.0;true;211;2;7528;0;0;0
2560.0;200.0;3270400.0;true;211;2;7713;0;0;0
512.0;250.0;3270400.0;true;211;2;7246;0;0;0
1024.0;250.0;3270400.0;true;211;2;7308;0;0;0
1536.0;250.0;3270400.0;true;211;2;7390;0;0;0
2048.0;250.0;3270400.0;true;211;2;7528;0;0;0
2560.0;250.0;3270400.0;true;211;2;7713;0;0;0
512.0;50.0;3924480.0;true;211;2;7246;0;0;0
1024.0;50.0;3924480.0;true;211;2;7308;0;0;0
1536.0;50.0;3924480.0;true;211;2;7390;0;0;0
2048.0;50.0;3924480.0;true;211;2;7528;0;0;0
2560.0;50.0;3924480.0;true;211;2;7713;0;0;0
512.0;100.0;3924480.0;true;211;2;7246;0;0;0
1024.0;100.0;3924480.0;true;211;2;7308;0;0;0
1536.0;100.0;3924480.0;true;211;2;7390;0;0;0
2048.0;100.0;3924480.0;true;211;2;7528;0;0;0
2560.0;100.0;3924480.0;true;211;2;7713;0;0;0
512.0;150.0;3924480.0;true;211;2;7246;0;0;0
1024.0;150.0;3924480.0;true;211;2;7308;0;0;0
1536.0;150.0;3924480.0;true;211;2;7390;0;0;0
2048.0;150.0;3924480.0;true;211;2;7528;0;0;0
2560.0;150.0;3924480.0;true;211;2;7713;0;0;0
512.0;200.0;3924480.0;true;211;2;7246;0;0;0
1024.0;200.0;3924480.0;true;211;2;7308;0;0;0
1536.0;200.0;3924480.0;true;211;2;7390;0;0;0
2048.0;200.0;3924480.0;true;211;2;7528;0;0;0
2560.0;200.0;3924480.0;true;211;2;7713;0;0;0
512.0;250.0;3924480.0;true;211;2;7246;0;0;0
1024.0;250.0;3924480.0;true;211;2;7308;0;0;0
1536.0;250.0;3924480.0;true;211;2;7390;0;0;0
2048.0;250.0;3924480.0;true;211;2;7528;0;0;0
2560.0;250.0;3924480.0;true;211;2;7713;0;0;0
"""
subset_name ="DurationII"#"Latency"
# Chargement des données dans un DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), delimiter=';')

# Création d'une figure avec deux sous-graphiques côte à côte
fig, axes = plt.subplots(1, 3, figsize=(15, 5))


# Liste de couleurs et styles pour distinguer les courbes
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
markers = ['o', 's', 'd', 'v', '^', '<', '>', 'p', 'h', '*']

# --- 1. Latency vs GRID_SIZE avec différentes valeurs de (NUM_MINOR_CYCLES, NUM_VISIBILITIES) ---
for (num_cycles, num_vis) in itertools.product(df["NUM_MINOR_CYCLES"].unique(), df["NUM_VISIBILITIES"].unique()):
    subset = df[(df["NUM_MINOR_CYCLES"] == num_cycles) & (df["NUM_VISIBILITIES"] == num_vis)]
    axes[0].plot(subset["GRID_SIZE"], subset[subset_name], 
                 marker=markers[(int(num_cycles) // 50) % len(markers)], 
                 linestyle='-', color=colors[(int(num_vis) // 1308160) % len(colors)], 
                 label=f"C={num_cycles}, V={num_vis}")

axes[0].set_xlabel("GRID_SIZE")
axes[0].set_ylabel("Latency")
#axes[0].set_title("Latence vs GRID_SIZE")
axes[0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
axes[0].grid()

# --- 2. Latency vs NUM_MINOR_CYCLES avec différentes valeurs de (GRID_SIZE, NUM_VISIBILITIES) ---
for (grid_size, num_vis) in itertools.product(df["GRID_SIZE"].unique(), df["NUM_VISIBILITIES"].unique()):
    subset = df[(df["GRID_SIZE"] == grid_size) & (df["NUM_VISIBILITIES"] == num_vis)]
    axes[1].plot(subset["NUM_MINOR_CYCLES"], subset[subset_name], 
                 marker=markers[(int(grid_size) // 512) % len(markers)], 
                 linestyle='-', color=colors[(int(num_vis) // 1308160) % len(colors)], 
                 label=f"G={grid_size}, V={num_vis}")

axes[1].set_xlabel("NUM_MINOR_CYCLES")
axes[1].set_ylabel("Latency")
#axes[1].set_title("Latence vs NUM_MINOR_CYCLES")
axes[1].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
axes[1].grid()

# --- 3. Latency vs NUM_VISIBILITIES avec différentes valeurs de (GRID_SIZE, NUM_MINOR_CYCLES) ---
for (grid_size, num_cycles) in itertools.product(df["GRID_SIZE"].unique(), df["NUM_MINOR_CYCLES"].unique()):
    subset = df[(df["GRID_SIZE"] == grid_size) & (df["NUM_MINOR_CYCLES"] == num_cycles)]
    axes[2].plot(subset["NUM_VISIBILITIES"], subset[subset_name], 
                 marker=markers[(int(grid_size) // 512) % len(markers)], 
                 linestyle='-', color=colors[(int(num_cycles) // 50) % len(colors)], 
                 label=f"G={grid_size}, C={num_cycles}")

axes[2].set_xlabel("NUM_VISIBILITIES")
axes[2].set_ylabel("Latency")
#axes[2].set_title("Latence vs NUM_VISIBILITIES")
axes[2].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1,1))
axes[2].grid()

# Titre global
fig.suptitle("G2G latency vs GRID_SIZE, NUM_MINOR_CYCLES, and NUM_VISIBILITIES", fontsize=16)

# Ajustement de l'affichage
plt.tight_layout()

# Enregistrer la figure en PNG
fig.savefig("simulation_g2g.png", format="png", dpi=400)

#plt.subplots_adjust(top=0.85, wspace=0.3)  # Augmenter l'espace horizontal entre les plots
plt.show()
