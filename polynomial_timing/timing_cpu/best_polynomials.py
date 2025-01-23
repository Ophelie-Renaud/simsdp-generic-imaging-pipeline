import os
import numpy as np
import pandas as pd
import math
import scipy.optimize
import matplotlib.pyplot as plt
import csv
# Dossiers
input_folder = "average"
output_folder = "polynomial_fits"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

def calculate_max_degree(num_points):
    """Calculer le degré maximum du polynôme basé sur le nombre de points."""
    return int((-3 + math.sqrt(9 + 8 * num_points)) // 2)
def load_data_and_axis(filename, num_axis):
    result = np.genfromtxt(filename, delimiter=",")
    div_by = num_axis + 1
    if len(result) % div_by != 0:
        print("Error: number of axis must be divisible by number of data points")
        exit()

    num_items = int(len(result) / div_by)
    return result.reshape((num_items, div_by))
def poly2Dgeneral(x, *coeffs):
    num_coeffs = len(coeffs)
    expected_coeffs = (dof+1) * (dof + 2) // 2

    assert(int(expected_coeffs) == int(num_coeffs))

    answer = 0
    curr_coeff_idx = 0

    for i in range(0, dof + 1):
        for j in range(0, dof - i + 1):
            curr_term = coeffs[curr_coeff_idx] * (x[0] ** i) * (x[1] ** j)
            answer += curr_term
            curr_coeff_idx += 1

    return answer
def fit2D(points_and_axes, dof, num_x, num_y):
    x1data, x2data = np.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y], indexing='ij')
    x1shape = x1data.shape
    x1data1d = x1data.reshape(np.prod(x1shape))
    x2data1d = x2data.reshape(np.prod(x1shape))
    axis_data = np.vstack((x1data1d, x2data1d))
    output_data = points_and_axes[0].flatten()

    num_coeffs = int((dof + 1) * (dof + 2) / 2)
    orig_guess = [0] * num_coeffs

    popt, pcov = scipy.optimize.curve_fit(poly2Dgeneral, axis_data, output_data, p0=orig_guess)

    return popt
def compute_mse_2D(data_points, coeffs, dof):
    rmse = 0
    total = 0

    vals = data_points[0].flatten()
    xs = data_points[1].flatten()
    ys = data_points[2].flatten()

    for i, data_point in enumerate(vals):
        val = poly2Dgeneral((xs[i], ys[i], dof), *coeffs)
        dif = val - data_point
        rmse += dif * dif
        total += 1

    rmse = math.sqrt(rmse)
    return rmse

# Parcourir les fichiers dans le dossier 'average'
for filename in os.listdir(input_folder):
    # Construire le chemin complet du fichier
    full_path = os.path.join(input_folder, filename)

    # Vérifier si c'est un fichier CSV
    if os.path.isfile(full_path) and filename.endswith(".csv") and filename ==  "clean_timings.csv":

        values = []
        param1 = []
        param2 = []
        # Lire les données du fichier
        with open(full_path, "r") as file:
            reader = csv.reader(file, delimiter=",")

            # Convertir toutes les lignes en une liste plate
            flattened_data = [value for row in reader for value in row]
            values = [flattened_data[i] for i in range(0, len(flattened_data), 3)]
            param1 = [flattened_data[i] for i in range(1, len(flattened_data), 3)]
            param2 = [flattened_data[i] for i in range(2, len(flattened_data), 3)]

        print(values)
        # Nombre de paramètres différents
        x = len(np.unique(param1))
        y = len(np.unique(param2))
        num_points = x * y

        data_points = load_data_and_axis(full_path, 1)
        points_and_axes = np.split(data_points, 2, axis=1)
        # Calculer le degré maximum
        max_degree = calculate_max_degree(num_points)

        best_rmse = float("inf")
        best_coeffs = None
        best_dof = None

        # Ajuster les polynômes
        for dof in range(1, max_degree + 1):
            coeffs = fit2D(points_and_axes, 1, len(param1)-1, len(param2)-1)
            rmse = compute_mse_2D(values, coeffs, dof)
            print(f"File: {filename}, Degree: {dof}, RMSE: {rmse}")
            if rmse < best_rmse:
                best_rmse = rmse
                best_coeffs = coeffs
                best_dof = dof
        print(f"File: {filename}, Best Degree: {best_dof}, Best RMSE: {best_rmse}")


            # Sauvegarde des résultats dans un fichier CSV
        output_csv = os.path.join(output_folder, "best_fit_results.csv")
        with open(output_csv, "w") as f:
            f.write("Filename,Best Degree,Best RMSE,Coefficients\n")
            #coeffs_str = ",".join(map(str, best_coeffs))
            f.write(f"{output_csv},{best_dof},{best_rmse},{best_coeffs}\n")

        print(f"Results saved to {output_csv}")
