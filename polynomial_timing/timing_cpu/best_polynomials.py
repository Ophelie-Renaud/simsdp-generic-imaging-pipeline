import os
import numpy as np
import pandas as pd
import math
import scipy.optimize
import matplotlib.pyplot as plt
import csv
"""
Script de calcul des fonctions fonction d'ajustement qui définie le temps d'exécution des calculs composant les pipeline d'imagerie de radio astronomie.

Ce script parcours une serie de polynome specifiant la fonction d'ajustement caractérisant le temps d'execution des calculs.
Les calculs sont parametré par 1 ou 2 paramètres (taille de grille, visibilité, etc.) resultant en un fonction d'ajustement d'une ou deux dimension. 
Le degré de polynome affect l'erreur entre le model et la mesure, ce scipt identifie le degré de polynome offrant l'erreur la plus faible pour chaque calculs.
Les résultats sont collectés et stocké pour une utilisation par le simulateur SimSDP pour simuler le déploiement de pipelines d'imagerie de radio astronomie et faciliter l'exploration algorithmique.

Fonctionnalités :
- ...

Auteur : Ophélie RENAUD
Date : 7/02/2025
"""


# Dossiers
input_folder = "average"
output_folder = "polynomial_fits"

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

def calculate_max_degree2D(num_points):
    """Calculer le degré maximum du polynôme basé sur le nombre de points."""
    return int((-3 + math.sqrt(9 + 8 * num_points)) // 2)
def calculate_max_degree1D(num_points):
    print("num_points")
    print(num_points)
    """Calculer le degré maximum du polynôme basé sur le nombre de points."""
    return int(num_points - 1)
def load_data_and_axis(filename, num_axis):
    result = np.genfromtxt(filename, delimiter=",")
    result = result[:-1]
    div_by = num_axis + 1
    if len(result) % div_by != 0:
        print("Error: number of axis %d must be divisible by number of data points %d",div_by, len(result), filename)
        exit()

    num_items = int(len(result) / div_by)
    return result.reshape((num_items, div_by))

def poly(x, coeffs):
    out = 0
    for deg, coeff in enumerate(coeffs[::-1]):
        out += coeff * (x ** deg)

    return out
def poly1Dgeneral(x, *coeffs):
    num_coeffs = len(coeffs)
    #print(coeffs)
    expected_coeffs = dof +1

    assert(int(expected_coeffs) == int(num_coeffs))

    answer = 0
    curr_coeff_idx = 0

    for i in range(0, dof + 1):
        curr_term = coeffs[curr_coeff_idx] * (x[0] ** i)
        answer += curr_term
        curr_coeff_idx += 1

    return answer
def poly2Dgeneral(x, *coeffs):
    num_coeffs = len(coeffs)
    expected_coeffs = (dof+1) * (dof + 2) // 2

    assert(int(expected_coeffs) == int(num_coeffs)), f"Error: Expected {expected_coeffs} coefficients, but got {num_coeffs}."

    answer = 0
    curr_coeff_idx = 0

    for i in range(0, dof + 1):
        for j in range(0, dof - i + 1):
            curr_term = coeffs[curr_coeff_idx] * (x[0] ** i) * (x[1] ** j)
            answer += curr_term
            curr_coeff_idx += 1

    return answer

def fit(points_and_axes, dof):
    points_and_axes[1].flatten()
    return np.polyfit(points_and_axes[1].flatten(), points_and_axes[0].flatten(), dof)
    
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

def plot1D(points_and_axes, coeffs = None, fitting_func = None):
    plt.figure()

    plt.subplot(1,1,1)
    plt.plot(points_and_axes[1].flatten(), points_and_axes[0].flatten(), 'ro')

    if coeffs is not None:
        x = np.linspace(0, points_and_axes[1].flatten()[-1])
        fitting_func = poly(x, coeffs)
        plt.plot(x, fitting_func, 'b-')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.show(block=False)

def plot2D(points_and_axes, coeffs, num_x, num_y):
    plt.figure()

    if coeffs is None:
        X1, X2 = np.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y])

        plt.subplot(1,1,1)
        plt.pcolormesh(X1, X2, points_and_axes[0].flatten().reshape((num_x, num_y)))
        plt.colorbar()

    elif coeffs is not None:
        X1, X2 = np.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y], indexing='ij')
        Z1 = points_and_axes[0].flatten().reshape((num_x, num_y))

        axes = plt.subplot(2,1,1)
        axes.set_xlim(points_and_axes[1][0], points_and_axes[1][-1])
        axes.set_ylim(points_and_axes[2][0], points_and_axes[2][-1])


        plt.pcolormesh(X1, X2, points_and_axes[0].flatten().reshape((num_x, num_y)), vmin = Z1.min(), vmax = Z1.max())
        plt.colorbar()

        xr = np.linspace(0, points_and_axes[1].flatten()[-1] * 2, num=100)
        yr = np.linspace(0, points_and_axes[2].flatten()[-1], num=100)

        X1, X2 = np.meshgrid(xr, yr, indexing='ij')
        size = X1.shape
        x1_1d = X1.reshape((1, np.prod(size)))
        x2_1d = X2.reshape((1, np.prod(size)))

        xdata = np.vstack((x1_1d, x2_1d))

        fitting_func = poly2Dgeneral(xdata, *coeffs)
        Z = fitting_func.reshape(size)

        axes = plt.subplot(2,1,2)
        axes.set_xlim(points_and_axes[1][0], points_and_axes[1][-1])
        axes.set_ylim(points_and_axes[2][0], points_and_axes[2][-1])
        plt.pcolormesh(X1, X2, Z, vmin = Z1.min(), vmax = Z1.max())
        plt.colorbar()

    plt.show(block=False)

def compute_mse_1D(data_points, coeffs, dof):
    rmse = 0
    total = 0

    vals = data_points[0].flatten()
    xs = data_points[1].flatten()

    for i, data_point in enumerate(vals):
        val = poly1Dgeneral((xs[i], dof), *coeffs)
        dif = val - data_point
        rmse += dif * dif
        total += 1

    rmse = math.sqrt(rmse)
    return rmse
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
def rmse(measure, predict):
    return np.sqrt(np.mean((np.array(measure) - np.array(predict))**2))
    
def print_poly1D(coeffs, var_name):
    terms = []
    degree = len(coeffs) - 1  # Déterminer le degré du polynôme

    for i, coef in enumerate(coeffs):
        power = degree - i  # Exposant du terme

        # Ignorer les coefficients nuls pour simplifier l'affichage
        if abs(coef) < 1e-7:
            continue

        # Formatage du coefficient
        coef_str = f"{coef:.6f}" if coef < 0 or i == 0 else f"+ {coef:.6f}"

        # Construction du terme
        if power == 0:
            terms.append(f"{coef_str}")  # Constante
        elif power == 1:
            terms.append(f"{coef_str} * {var_name}")  # Terme linéaire
        else:
            terms.append(f"{coef_str} * {var_name}^{power}")  # Terme polynomial

    # Assemblage final et affichage
    polynomial = " ".join(terms)
    return polynomial

def print_poly2D(coeffs, var_name1,var_name2, dof):
    terms = []
    index = 0

    for i in range(dof + 1):  # Parcourt les degrés de x
        for j in range(i + 1):   # Parcourt les degrés de y
            coef = coeffs[index]
            index += 1

            if abs(coef) < 1e-7:  # Ignore les coefficients trop petits
                continue

            # Formatage du coefficient
            coef_str = f"{coef:.6f}" if coef < 0 or not terms else f"+ {coef:.6f}"

            # Gestion des puissances
            term = coef_str
            if i - j > 0:
                term += f" * {var_name1}^{i - j}" if (i - j) > 1 else f" * {var_name1}"
            if j > 0:
                term += f" * {var_name2}^{j}" if j > 1 else f" * {var_name2}"

            terms.append(term)

    # Assemblage final
    polynomial = " ".join(terms) if terms else "0"
    return polynomial

# Sauvegarde des résultats dans un fichier CSV
output_csv = os.path.join(output_folder, "best_fit_results.csv")
with open(output_csv, "w") as f:
    f.write("Filename,Best Degree,Best RMSE,Coefficients,Average\n")

tests = [
    ("addvis_timings", 1,["NUM_VISIBILITIES"]), #int NUM_VISIBILITIES (func= poly1D)
    ("clean_timings", 2, ["GRID_SIZE","NUM_MINOR_CYCLES"]), #int GRID_SIZE, int NUM_MINOR_CYCLES (func= poly2D)
    ("config_sequel_timings", 0,[]), #None (func=const)
    ("config_timings", 0,[]), #None (func=const)
    ("convolution_correction_actor_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)

    ("correction_setup_timings", 1,["GRID_SIZE"]),#int GRID_SIZE (func= poly1D)
    ("correct_to_finegrid_timings", 1,["NUM_VISIBILITIES"]), #int NUM_VISIBILITIES (func= poly1D)

    ("degrid_timings", 2,["GRID_SIZE","NUM_VISIBILITIES"]), #int GRID_SIZE, int NUM_VISIBILITIES (func= poly2D)
    ("dft_timings", 2,["NUM_MAX_SOURCES","NUM_VISIBILITIES"]), #int NUM_MINOR_CYCLES, int NUM_VISIBILITIES (func= poly2D)
    ("dgkernel_timings", 0,[]),#None (func=const)

    ("fft_shift_complex_to_complex_actor_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)
    ("fft_shift_complex_to_real_actor_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)
    ("fft_shift_real_to_complex_actor_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)

    ("CUFFT_EXECUTE_FORWARD_C2C_actor_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)

    ("gains_apply_timings", 1,["NUM_VISIBILITIES"]), #int NUM_VISIBILITIES (func= poly1D)
    ("gkernel_timings", 0,[]), #None (func=const)
    #("grid_timings", 2,["GRID_SIZE","NUM_VISIBILITIES"]), #int GRID_SIZE, int NUM_VISIBILITIES (func= poly2D)

    ("psf_host_set_up_timings", 1,["GRID_SIZE"]),#int GRID_SIZE (func= poly1D)
    ("reciprocal_transform_timings", 1,["NUM_VISIBILITIES"]), #int NUM_VISIBILITIES (func= poly1D)



    ("s2s_timings", 2,["GRID_SIZE","NUM_VISIBILITIES"]), #int GRID_SIZE, int NUM_VISIBILITIES (func= poly2D)
    ("save_output_timings", 1,["GRID_SIZE"]), #int GRID_SIZE (func= poly1D)

    ("subtract_from_measurements_timings", 1,["NUM_VISIBILITIES"]),#int NUM_VISIBILITIES (func= poly1D)
    ("subtraction_imagespace_timings", 1,["GRID_SIZE"]),#int GRID_SIZE (func= poly1D)
]
# Parcourir les fichiers dans le dossier 'average'
for test in tests:
    function_name, num_axis, name_axis = test

    # Construire le chemin complet du fichier
    full_path = os.path.join(input_folder, function_name+".csv")

    data_points = load_data_and_axis(full_path, num_axis)

    best_rmse = float('inf')
    best_coeffs = None
    best_dof = None
    average_mes = None
    rmse_value = 0
    poly_str = None

    if num_axis== 0:
        #no parameter no fit
        best_rmse = 0
        best_coeffs = 0
        best_dof = 0
        average_mes = 0

    elif num_axis == 1:
        #split data_points into 2 section (value and part_x)
        points_and_axes = np.split(data_points, 2, axis=1)

        part_val = points_and_axes[0]
        average_mes = np.mean(part_val)

        part_x = points_and_axes[1]
        num_x = len(np.unique(part_x[:, 0]))

        # Calculer le degré maximum
        max_degree = calculate_max_degree1D(num_x)

        #rmse_value = 0
        #best_rmse = 0
        for dof in range(1, max_degree + 1):
            if dof > 0:
                coeffs = fit(points_and_axes, dof)
                #rmse = compute_mse_1D(points_and_axes, coeffs, dof)
                measure = points_and_axes[0]
                predict = poly(points_and_axes[1],coeffs)
                rmse_value = rmse(measure, predict)
            else:
                coeffs = 0
                rmse_value = 0

            #print("RMSE: " + str(rmse_value))
            #print(coeffs)

            if rmse_value < best_rmse:
                best_rmse = rmse_value
                best_coeffs = coeffs
                best_dof = dof
                

        poly_str = "max(0,"+print_poly1D(best_coeffs,name_axis[0])+")"
        print(poly_str)
        #dof = best_dof
        #plot1D(points_and_axes, best_coeffs)


    elif num_axis == 2:
        #split data_points into 3 section (value, part_x and part_y)
        points_and_axes = np.split(data_points, 3, axis=1)

        part_val = points_and_axes[0]
        average_mes = np.mean(part_val)

        part_x = points_and_axes[1]
        part_y = points_and_axes[2]

        num_x = len(np.unique(part_x[:, 0]))
        num_y = len(np.unique(part_y[:, 0]))

        max_degree = calculate_max_degree2D(num_x*num_y)

        #rmse_value = 0
        #best_rmse = 0
        for dof in range(1, max_degree):
            coeffs = fit2D(points_and_axes, dof, num_x, num_y)
            rmse_value = compute_mse_2D(points_and_axes, coeffs, dof)

            print("RMSE: " + str(rmse_value))
            print(coeffs)
            if rmse_value < best_rmse:
                best_rmse = rmse_value
                best_coeffs = coeffs
                best_dof = dof
            #plot2D(points_and_axes, coeffs, num_x, num_y)
            #if dof == max_degree :
                #i don't why I can't the best coef
            #dof = best_dof
#            plot2D(points_and_axes, best_coeffs, num_x, num_y)

        poly_str = "max(0,"+ print_poly2D(best_coeffs,name_axis[0],name_axis[1],best_dof)+")"
        print(poly_str)

    else:
        print("Error: dimensions not 1 or 2 are currently not supported")


    print(f"File: {test}, Best Degree: {best_dof}, Best RMSE: {best_rmse}")


        # Sauvegarde des résultats dans un fichier CSV
    output_csv = os.path.join(output_folder, "best_fit_results.csv")
    with open(output_csv, "a") as f:
        #coeffs_str = ",".join(map(str, best_coeffs))
        f.write(f"{function_name},{best_dof},{best_rmse},{poly_str},{average_mes}\n")

    print(f"Results saved to {output_csv}")
# À la fin, toutes les fenêtres resteront ouvertes
#plt.show()  # Ce `show` final permet de garder les fenêtres ouvertes
