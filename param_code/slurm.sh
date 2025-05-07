#!/bin/bash

#SBATCH --job-name=ri
#SBATCH --output=%x.o%j 
#SBATCH --time=00:20:00 # execution time max = 20 min
#SBATCH --nodes=2 #from 1 to 216? nodes available on ruche mesocentre
#SBATCH --ntasks-per-node=1 # mpi processes per node
#SBATCH --partition=cpu_short

# Load necessary modules
module purge
module load intel/19.0.3/gcc-4.8.5
#module load singularity/3.5.3/gcc-11.2.0


# Run executable
./SEP_Pipeline 1 1 1

# Run singularity image
mpirun - n 2 singularity run sdp_pipeline.sif dft 1 1 1 1 0000.ms #
