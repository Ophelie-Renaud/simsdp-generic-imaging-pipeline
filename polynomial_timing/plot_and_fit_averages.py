import numpy
import sys
import argparse
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import math

def load_data_and_axis(filename, num_axis):
	result = numpy.genfromtxt(filename, delimiter=",")
	div_by = num_axis + 1
	if len(result) % div_by != 0:
		print("Error: number of axis must be divisible by number of data points")
		exit()

	num_items = int(len(result) / div_by)
	return result.reshape((num_items, div_by))

def poly(x, coeffs):
	out = 0
	for deg, coeff in enumerate(coeffs[::-1]):
		out += coeff * (x ** deg)

	return out

def poly2D(x, a, b, c, d, e, f):
	return a * x[0] * x[0] + b * x[1] * x[1] + c * x[0] * x[1] + d * x[0] + e * x[1] + f

dof = 0

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
	return numpy.polyfit(points_and_axes[1].flatten(), points_and_axes[0].flatten(), dof)

def fit2D(points_and_axes, dof, num_x, num_y):
	x1data, x2data = numpy.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y], indexing='ij')
	x1shape = x1data.shape
	x1data1d = x1data.reshape(numpy.prod(x1shape))
	x2data1d = x2data.reshape(numpy.prod(x1shape))
	axis_data = numpy.vstack((x1data1d, x2data1d))
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
		x = numpy.linspace(0, points_and_axes[1].flatten()[-1])
		fitting_func = poly(x, coeffs)
		plt.plot(x, fitting_func, 'b-')

	plt.xlabel('x')
	plt.ylabel('y')

	plt.show()

def plot2D(points_and_axes, coeffs, num_x, num_y):
	plt.figure()

	if coeffs is None:
		X1, X2 = numpy.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y])

		plt.subplot(1,1,1)
		plt.pcolormesh(X1, X2, points_and_axes[0].flatten().reshape((num_x, num_y)))
		plt.colorbar()

	elif coeffs is not None:
		X1, X2 = numpy.meshgrid(points_and_axes[1].flatten()[0::num_y], points_and_axes[2].flatten()[0:num_y], indexing='ij')
		Z1 = points_and_axes[0].flatten().reshape((num_x, num_y)) 

		axes = plt.subplot(2,1,1)
		axes.set_xlim(points_and_axes[1][0], points_and_axes[1][-1])
		axes.set_ylim(points_and_axes[2][0], points_and_axes[2][-1])


		plt.pcolormesh(X1, X2, points_and_axes[0].flatten().reshape((num_x, num_y)), vmin = Z1.min(), vmax = Z1.max())
		plt.colorbar()

		xr = numpy.linspace(0, points_and_axes[1].flatten()[-1] * 2, num=100)
		yr = numpy.linspace(0, points_and_axes[2].flatten()[-1], num=100)

		X1, X2 = numpy.meshgrid(xr, yr, indexing='ij')
		size = X1.shape
		x1_1d = X1.reshape((1, numpy.prod(size)))
		x2_1d = X2.reshape((1, numpy.prod(size)))

		xdata = numpy.vstack((x1_1d, x2_1d))

		fitting_func = poly2Dgeneral(xdata, *coeffs)
		Z = fitting_func.reshape(size)

		axes = plt.subplot(2,1,2)
		axes.set_xlim(points_and_axes[1][0], points_and_axes[1][-1])
		axes.set_ylim(points_and_axes[2][0], points_and_axes[2][-1])
		plt.pcolormesh(X1, X2, Z, vmin = Z1.min(), vmax = Z1.max())
		plt.colorbar()

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


num_axis = int(sys.argv[2])
filename = sys.argv[1]
dof = int(sys.argv[3])

if num_axis == 2:
	num_x = int(sys.argv[4])
	num_y = int(sys.argv[5])


data_points = load_data_and_axis(filename, num_axis)


if num_axis == 1:
	points_and_axes = numpy.split(data_points, 2, axis=1)
	if dof > 0:
		coeffs = fit(points_and_axes, dof)
		plot1D(points_and_axes, coeffs)
		print(coeffs)
	else:
		plot1D(points_and_axes)

elif num_axis == 2:
	points_and_axes = numpy.split(data_points, 3, axis=1)
	coeffs = fit2D(points_and_axes, dof, num_x, num_y)
	print("RMSE: " + str(compute_mse_2D(points_and_axes, coeffs, dof)))
	print(coeffs)
	#print(poly2Dgeneral((2048*2048, 3924480), *coeffs))
	plot2D(points_and_axes, coeffs, num_x, num_y)
else:
	print("Error: dimensions not 1 or 2 are currently not supported")

#fitting here
