import numpy
import sys
import argparse
import csv

def compute_average_from_csv(filename):
	result = numpy.genfromtxt(filename, delimiter=",")
	av = numpy.mean(result)
	return av

def flatten(lst):
	output = []
	for val in lst:
		if type(val) is list:
			output += flatten(val)
		else:
			output += [val]

	return output

CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--files",
  nargs="*",
  type=str,
  default=["test"],
)
CLI.add_argument(
  "--x",
  nargs="*",
  type=float,
  default=[0],
)
CLI.add_argument(
  "--y",
  nargs="*",
  type=float,
  default=[],
)
CLI.add_argument(
  "--output",
  nargs=1,
  type=str,
  default=["output.csv"],
)

args = CLI.parse_args()



zipped = None

if len(args.y) == 0:
	if len(args.files) != len(args.x):
		print("Error: Number of files specified must be equal to the number of x-axis labels")
		quit()

	zipped = zip(args.files, args.x)
else:
	vals = []
	for xval in args.x:
		for yval in args.y:
			vals += [[xval, yval]]

	if len(args.files) != len(vals):
		print("Error: Number of files specified must be equal to the number of axis label combinations")
		quit()

	zipped = zip(args.files, vals)

output = []
for filename, axis_val in zipped:
	average = compute_average_from_csv(filename)
	output += flatten([average, axis_val])

out = open(args.output[0], 'w')
writer = csv.writer(out)
writer.writerow(output)
out.close()