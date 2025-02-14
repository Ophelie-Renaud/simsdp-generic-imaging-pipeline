import matplotlib.pyplot as plt
import numpy as np
import random

# Données de la SOTA
sota_labels = ["addvis", "clean", "degrid", "dft", "dgkernel", "fft", "fftshift", "finegrid", "gains_apply", "gkernel", "grid", "prolate", "prolate_setup", "s2s", "save_output", "sub_ispace", "vis_load"]
sota_rmse = [0.26, 1722.01, 1310.35, 328.79, None, 85.37, 0.21, 0.55, 1.81, None, 1311.53, 0.31, 0.0, 1729.34, 30.29, 0.21, 6.69]

# RMSE depuis une regression polynomial avec critère d'optimisation 
my_rmse = [0.26, 58, 10, 10, None, 0.01, 0.01, 0.01, 0.01, None, 10, 0.01, 0.01, 10, 0.01, 0.01, 6.69]

# Valeurs moyennes mesurées
average_measure_wang = [42, 9990, 53963, 22174, 58, 459, 204, 272, 1636, 53, 2890, 211, 0, 3961, 5794, 13, 12223]

average_measure_us = [420, 99900, 539630, 221740, 580, 4590, 2040, 2720, 16360, 530, 28900, 2110, 0, 39610, 57940, 130, 122230]

# Filtrer les valeurs valides
indices = [i for i, rmse in enumerate(sota_rmse) if rmse is not None]
labels = [sota_labels[i] for i in indices]
sota_valid = [sota_rmse[i] for i in indices]
my_valid = [my_rmse[i] for i in indices]
avg_valid_wang = [average_measure_wang[i] for i in indices]
avg_valid_us = [average_measure_us[i] for i in indices]

# Paramètres d'affichage
x = np.arange(len(labels))
thresh = 100  # Seuil de succès

# Création du graphique
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.3  # Largeur des barres

# Barres pour la SOTA
ax.bar(x - width, sota_valid, width, label='S. Wang et. al', color = '#4B0082')

# Barres pour ton algo
ax.bar(x, my_valid, width, label='Proposed Model',color = '#FF6B6B')

# Courbe des valeurs moyennes mesurées
ax.plot(x, avg_valid_wang, marker='o', linestyle='-', color='#4B0082', label='Average Measured Value S. Wang et. al')
ax.plot(x, avg_valid_us, marker='o', linestyle='-', color='#FF6B6B', label='Average Measured Value us')

# Labels et légende
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=14)
ax.set_ylabel("RMSE measured vs. predicted data", fontsize=14)
#ax.set_title("Comparaison des RMSE entre la SOTA et mon algorithme avec valeurs moyennes")
ax.legend(fontsize=14)
ax.set_yscale('log')

# Enregistrer la figure en PNG
fig.savefig("pic/comparaison_rmse.png", format="png", dpi=300)

# Affichage
plt.tight_layout()
plt.show()

