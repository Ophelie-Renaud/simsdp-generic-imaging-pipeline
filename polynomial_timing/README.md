# Automatic calculation of fitting function

The former method for calculating fitting function is tedious and time consuming to setting up. In order to automate the process to be able to run larger sampling and evaluate additional configuration some scripts have been implemented.

The proposed method consist in two automated steps:

- The first step consist in building the benchmark of timing for each dataflow actor.
- The second step consist in building fitting functions for each actor and store the one that give the best RMSE among all tested configurations.

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/poly_fit.png)

## Run the former method

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

`python plot_and_fit_averages.py averages/degrid.csv 2 1 8 4`

</details>

## Run the automatic method for CPU timing

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

`cd timing_cpu` > `cmake .` > `make`

It will generate a `SEP_Pipeline` executable.

Run: `python timing_benchmark_generation.py`

This will save the timing for each actor evaluated with the different parameter values and sample sizes specified in the python script in a `actorname_timings` file.

Run: `python best_polynomials.py`

This will compute polynomials from the /average folder and save the polynomial providing the best RMSE in the /polynimials_fits folder. 

</details>

## Run the automatic method for GPU timing

For sure like me you don't have NVIDIA GPU on your laptop. First of all: shame on us. Second of all here are how I obtain my result with the on Grid5000 cluster and on Ruche Mesocentre:

#### Grid5000 cluster
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

```
#copy file
scp -r timing_cpu orenaud@access.grid5000.fr:rennes
```

```
#connect the cluster
ssh orenaud@access.grid5000.fr
ssh rennes

#take a node with NVIDIA GPU
oarsub -q besteffort -p abacus19 -I

#check if there are nvidia
lspci | grep -i nvidia
nvcc -V
```
Each time you change of node:
```
#install all required lib
sudo-g5k apt-get install libfftw3-dev

#BLAS
sudo-g5k apt-get install libblas-dev

#LAPACK
sudo-g5k apt-get install liblapack-dev
sudo-g5k apt-get install liblapacke-dev

#notebook to visualize
sudo-g5k apt-get install python3-pip
sudo-g5k apt install jupyter-notebook

#ASTROPY
sudo-g5k apt install python3-astropy
```
Run the code:
```
cd timing_gpu
cmake .
make
./sep
```
From here the steps are the same as on CPU:
`cd timing_cpu` > `cmake .` > `make`
...
</details>

#### Ruche Mesocentre
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

```
#copy file
scp -r Code renaudo@ruche.mesocentre.universite-paris-saclay.fr:/workdir/renaudo
```
```
#connect the cluster
ssh renaudo@ruche.mesocentre.universite-paris-saclay.fr
#retrieve your files
cd /workdir/renaudo/Code
```
Construct your slurm script:
```
touch job_timing.sh

nano job_timing.sh
```

```slurm
#!/bin/bash
#SBATCH --job-name=sep_c      # Nom du job
#SBATCH --output=output_%j.txt    # Nom du fichier de sortie 
#SBATCH --error=error_%j.txt      # Nom du fichier d'erreur
#SBATCH --ntasks=1                # Nombre de tâches à exécuter
#SBATCH --cpus-per-task=4         # Nombre de cœurs par tâche
#SBATCH --time=01:00:00          
#SBATCH --partition=gpu      

# Chemin vers l'exécutable 
executable=./SEP_pipeline       

# Execution de l'exécutable
srun $executable
```
Run the job:
```
sbatch job_timing.sh
```
</details>

Once the timings are computed, on the PREESM projects e.g:  simsdp-g2g-imaging-pipeline/timings.csv replace the timing by the computed formula.

## References

*[r] S. Wang, N. Gac, H. Miomandre, J.-F. Nezan, K. Desnos, F. Orieux « An Initial Framework for Prototyping Radio-Interferometric Imaging Pipelines»*
