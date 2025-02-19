# Automatic calculation of fitting function

The former method for calculating fitting function is tedious and time consuming to setting up. In order to automate the process to be able to run larger sampling and evaluate additional configuration some scripts have been implemented.

The proposed method consist in two automated steps:

- The first step consist in building the benchmark of timing for each dataflow actor.
- The second step consist in building fitting functions for each actor and store the one that give the best RMSE among all tested configurations.

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/pic/poly_fit.png)

## SOTA [*S. Wang, et al*](#ref)

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>


All the steps are detail on the main readme and you have to use this command for each averages/timing files:
Considering that computation execution time have been manually processed and stored in the  `average` (`cd polynomial_timing/sota/`) folder then run the following command:

> [!TIP]
>
> Do not run script from CLion, it will crash if you don't set enough resource, preferred run from prompt.

`python plot_and_fit_averages.py averages/addvis.csv 1 1 5 0`  RMSE = 0.2607680962081062  :white_check_mark:

`python plot_and_fit_averages.py averages/clean.csv 2 2 4 4`  RMSE = 1722.0090647527306 :x:

`python plot_and_fit_averages.py averages/degrid.csv 2 1 8 4`  RMSE = 1310.353129574978 :x:

`python plot_and_fit_averages.py averages/dft.csv 2 2 4 4`  RMSE = 328.79013774421367 :x:

> Bottleneck parameters: NUM_MINOR_CYCLE, NUM_VIS

`python plot_and_fit_averages.py averages/dgkernel.csv 1 0 1 0`  RMSE = NA :x:

`python plot_and_fit_averages.py averages/fft.csv 1 1 5 0`  RMSE = 85.36911021588533 :white_check_mark:

> Bottleneck parameters: GRID_SIZE

`python plot_and_fit_averages.py averages/fftshift.csv 1 1 5 0`  RMSE = 0.2105075974340459 :white_check_mark:

`python plot_and_fit_averages.py averages/finegrid.csv 1 1 5 0`  RMSE = 0.5555977761734401 :white_check_mark:

`python plot_and_fit_averages.py averages/gains_apply.csv 1 1 5 0`  RMSE = 1.81580101699881 :white_check_mark:

`python plot_and_fit_averages.py averages/gkernel.csv 1 0 1 0`  RMSE = NA :x:

`python plot_and_fit_averages.py averages/grid.csv 2 2 8 4`  RMSE = 1311.535059598015 :x:

`python plot_and_fit_averages.py averages/prolate.csv 1 1 5 0`  RMSE = 0.31412379399377993​ :white_check_mark:

`python plot_and_fit_averages.py averages/prolate_setup.csv 1 1 5 0`  RMSE = 0.0 :white_check_mark:

`python plot_and_fit_averages.py averages/s2s.csv 2 2 8 4`  RMSE = 1729.33617134492 :x:

> Bottleneck parameters: GRID_SIZE, NUM_VIS

`python plot_and_fit_averages.py averages/save_output.csv 1 1 5 0`  RMSE = 30.2903978986482  :x:

`python plot_and_fit_averages.py averages/sub_ispace.csv 1 1 5 0`  RMSE = 0.21088577379699533 :white_check_mark:

`python plot_and_fit_averages.py averages/vis_load.csv 1 1 5 0`  RMSE = 6.689979239296699 :white_check_mark:

The RMSE alone does not allow conclusions to be drawn about the model's performance, but it does provide some insights. The fewer parameters required for the calculations, the simpler the model, the more reliable the adjustment function.
Calculations depending on 2 parameters are the most difficult to model, 2-dimensional polynomials with a degree of 2 are not reliable enough to model their calculation time. Hence the need to evaluate several possible fitting functions.

<!--`python plot_and_fit_averages.py averages/degrid.csv 2 1 8 4`-->

<!--If you want to use the same command for the automated generated files there is a bug that you can bypass inserting :-->

<!--
def load_data_and_axis(filename, num_axis):
	result = numpy.genfromtxt(filename, delimiter=",")
	result = result[:-1] #<-- this line 
-->

</details>

## Proposed method (automated polynomial regression with RMSE as optimization criteria) - on your laptop :computer:

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

`cd timing_cpu` > `cmake .` > `make`

It will generate a `SEP_Pipeline` executable.

Run: `python timing_benchmark_generation.py`

This will save the timing for each actor evaluated with the different parameter values and sample sizes specified in the python script in a `actorname_timings` file.

Run: `python best_polynomials.py`

This will compute polynomials from the /average folder and save the polynomial providing the best RMSE in the /polynimials_fits folder. 

| 📝 **Note**                                                   |
| ------------------------------------------------------------ |
| The process is the same for GPU from the 📁 `timing_gpu`  if your laptop is equipped with NVIDIA GPU. |

</details>

Here are the result comparing the RMSE between measured values and model result (the manual model vs. our proposed model) for each computation:

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/pic/comparaison_rmse.png)

The average measured value are here just to put RMSE into perspective. Our average measured value are greater than the SOTA since we browse larger range of configuration. Our proposed method allow equals RMSE on simple computation (computation with 1 parameter and best polynomial is linear (deg = 1)) and lower RMSE on complex computation since we browse several polynomials and the degree of polynomial is greater because our samples are larger. However comparing RMSE on the training measures is not enough to evaluate the performance of our model, to do so we should compare the simulated latency (where computation timing is the fitting function) to the measured latency running the full pipeline.

## Proposed method (automated polynomial regression with RMSE as optimization criteria) - on remote cluster

| 📝 **Note**                                                   |
| ------------------------------------------------------------ |
| The **ongoing work** gives an alternative to run the code on external resources. I m trying to exhibit how I obtain results on `Grid5000 cluster` and on `Ruche Mesocentre` 🗄️. |

#### Grid5000 cluster

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

Here are the following step to run on Grid5000 cluster: 

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

```bash
#copy file
scp -r timing_cpu orenaud@access.grid5000.fr:rennes
```

```bash
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
```bash
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
```bash
cd timing_gpu
cmake .
make
./sep
```
From here the steps are the same as on CPU:
`cd timing_cpu` > `cmake .` > `make`
...

</details>

Otherwise a script is provided to automatically transfer files, connect to the required node and submit a job that will execute the script: `python3 run_on_grid5000.py`.


</details>

#### Ruche Mesocentre
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

```bash
#copy file
scp -r Code renaudo@ruche.mesocentre.universite-paris-saclay.fr:/workdir/renaudo
```
```bash
#connect the cluster
ssh renaudo@ruche.mesocentre.universite-paris-saclay.fr
#retrieve your files
cd /workdir/renaudo/Code
```
Construct your slurm script:
```bash
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

> [!WARNING]
>
> Once the timings are computed, on the PREESM projects e.g:  simsdp-g2g-imaging-pipeline/timings.csv replace the timing by the computed formula.

## References

<a id="ref"></a> [Manual regression & 1 node generic imaging pipeline](https://hal.science/hal-04361151/file/paper_dasip24_5_wang_updated-2.pdf): S. Wang, N. Gac, H. Miomandre, J.-F. Nezan, K. Desnos, F. Orieux « An Initial Framework for Prototyping Radio-Interferometric Imaging Pipelines».

