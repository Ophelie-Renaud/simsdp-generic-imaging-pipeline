import matplotlib.pyplot as plt
import numpy as np
import random

# Données de la SOTA
sota_labels = ["addvis", "clean", "degrid", "dft", "dgkernel", "fft", "fftshift", "finegrid", "gains_apply", "gkernel", "grid", "prolate", "prolate_setup", "s2s", "save_output", "sub_ispace", "vis_load"]
sota_rmse = [0.26, 1722.01, 1310.35, 328.79, None, 85.37, 0.21, 0.55, 1.81, None, 1311.53, 0.31, 0.0, 1729.34, 30.29, 0.21, 6.69]

# Simuler des RMSE pour ton algo avec des valeurs aléatoires
my_rmse = [rmse * random.uniform(0.8, 1.2) if rmse is not None else None for rmse in sota_rmse]

# Valeurs moyennes mesurées
average_measure = [42, 9990, 53963, 22174, 58, 459, 204, 272, 1636, 53, 2890, 211, 0, 3961, 5794, 13, 12223]

# Filtrer les valeurs valides
indices = [i for i, rmse in enumerate(sota_rmse) if rmse is not None]
labels = [sota_labels[i] for i in indices]
sota_valid = [sota_rmse[i] for i in indices]
my_valid = [my_rmse[i] for i in indices]
avg_valid = [average_measure[i] for i in indices]

# Paramètres d'affichage
x = np.arange(len(labels))
thresh = 100  # Seuil de succès

# Création du graphique
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.3  # Largeur des barres

# Barres pour la SOTA
ax.bar(x - width, sota_valid, width, label='S. Wang et. al')

# Barres pour ton algo
ax.bar(x, my_valid, width, label='Proposed Model')

# Courbe des valeurs moyennes mesurées
ax.plot(x, avg_valid, marker='o', linestyle='-', color='black', label='Average Measured Value')

# Labels et légende
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
ax.set_ylabel("RMSE measured vs. predicted data", fontsize=14)
#ax.set_title("Comparaison des RMSE entre la SOTA et mon algorithme avec valeurs moyennes")
ax.legend(fontsize=14)
ax.set_yscale('log')

# Enregistrer la figure en PNG
fig.savefig("comparaison_rmse.png", format="png", dpi=300)

# Affichage
plt.tight_layout()
plt.show()

