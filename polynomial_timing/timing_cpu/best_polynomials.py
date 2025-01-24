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
        print("Error: number of axis %d must be divisible by number of data points %d",div_by, len(result), filename)
        exit()

    num_items = int(len(result) / div_by)
    return result.reshape((num_items, div_by))

def poly(x, coeffs):
    out = 0
    for deg, coeff in enumerate(coeffs[::-1]):
        out += coeff * (x ** deg)

    return out
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

    plt.show()
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


tests = [
    ("addvis_timings", 1), #int NUM_SAMPLES
    ("clean_timings", 2), #int NUM_SAMPLES, int GRID_SIZE
    ("config_sequel_timings", 0), #int NUM_SAMPLES, int NUM_VISIBILITIES
    ("config_timings", 0), #int NUM_SAMPLES, int GRID_SIZE
    ("correct_to_finegrid_timings", 1), #int NUM_SAMPLES, int NUM_MINOR_CYCLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("dft_timings", 2), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("dgkernel_timings", 0), #int NUM_SAMPLES, int NUM_VISIBILITIES
    ("fftshift_timings", 1), #int NUM_SAMPLES, int GRID_SIZE
    ("fft_timings", 1), #int NUM_SAMPLES, int NUM_VISIBILITIES, int NUM_ACTUAL_VISIBILITIES
    ("gains_apply_timings", 1), #int NUM_SAMPLES, int GRID_SIZE
    ("gains_reciprocal_transform_timings", 1), #int NUM_SAMPLES, int GRID_SIZE
    ("gkernel_timings", 0), #int NUM_SAMPLES, int GRID_SIZE
    ("prolate_setup_timings", 1),
    ("prolate_timings", 1),
    ("save_output_timings", 1),
    ("subtraction_imagespace_timings", 1),# int NUM_SAMPLES, int GRID_SIZE, int NUM_MINOR_CYCLES
]
# Parcourir les fichiers dans le dossier 'average'
for test in tests:
    function_name, num_axis = test
    #num_axis=num_axis-1
    # Construire le chemin complet du fichier
    full_path = os.path.join(input_folder, function_name+".csv")

    data_points = load_data_and_axis(full_path, num_axis)
    print(num_axis)
    print(data_points)

    best_rmse = float("inf")
    best_coeffs = None
    best_dof = None

    if num_axis == 1:
        points_and_axes = np.split(data_points, 2, axis=1)
        print(points_and_axes)
        num_x = len(points_and_axes[1])
        print(num_x)
        # Calculer le degré maximum
        max_degree = calculate_max_degree(num_x)
        for dof in range(1, max_degree + 1):
            if dof > 0:
                coeffs = fit(points_and_axes, dof)
                #plot1D(points_and_axes, coeffs)
                rmse = 0
            else:
                #plot1D(points_and_axes)
                rmse = 0
            if rmse < best_rmse:
                best_rmse = rmse
                best_coeffs = coeffs
                best_dof = dof

    elif num_axis == 2:
        points_and_axes = np.split(data_points, 3, axis=1)
        # Calculer le degré maximum
        max_degree = calculate_max_degree(points_and_axes)
        num_x = 1
        num_y = 1

        for dof in range(1, max_degree + 1):
            coeffs = fit2D(points_and_axes, dof, num_x, num_y)
            rmse = compute_mse_2D(points_and_axes, coeffs, dof)
            #print(poly2Dgeneral((2048*2048, 3924480), *coeffs))
            #plot2D(points_and_axes, coeffs, num_x, num_y)
    else:
        print("Error: dimensions not 1 or 2 are currently not supported")


    print(f"File: {test}, Best Degree: {best_dof}, Best RMSE: {best_rmse}")


        # Sauvegarde des résultats dans un fichier CSV
    output_csv = os.path.join(output_folder, "best_fit_results.csv")
    with open(output_csv, "w") as f:
        f.write("Filename,Best Degree,Best RMSE,Coefficients\n")
        #coeffs_str = ",".join(map(str, best_coeffs))
        f.write(f"{output_csv},{best_dof},{best_rmse},{best_coeffs}\n")

    print(f"Results saved to {output_csv}")
