# SimSDP - Generic Imaging Pipeline

SimSDP is an HPC resource allocation framework and simulation tool designed to optimize intra-node and inter-node resource allocation for scientific data processing. It combines **PREESM** for rapid prototyping and **SimGrid** for simulating inter-node communication and computation. 

The primary goal of SimSDP is to simulate the **Science Data Processor (SDP)** pipeline from the Square Kilometre Array (SKA), converting visibilities into output images on massive HPC architectures, handling datasets as large as the SKA requirements (xTB).


The aim of this project is to facilitate the comparison of algorithm describing the SDP using SimSDP.  As a dataflow-based tool, SimSDP takes into account the dataflow application graph as well as a csv file containing the execution times of the actors on each target to be simulated. Dataflow representation of  the **Generic Imaging Pipeline** with a single and multi-frequency scenarii are provided. Our goal is to simulate these pipelines on Multinode - Multicore architecture, on Multinode - MonoGPU architecture and (might be) on Multinode - MultiGPU architecture.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal.png" style="zoom:100%;" />

Algorithm performance is influenced by parameters such as the number of visibility points, grid size and number of minor cycles. We aim to identify and evaluate the influence of these parameters on scaling and assess whether a solution is viable in the context of SKA.

To achieve this, we provide tools that automatically estimate the execution time of dataflow actors for each targeted architecture and simulate algorithm performance by varying a set of parameter values automatically.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal2.png" style="zoom:100%;" />

---

## Repository Structure

This repository contains six main directories, categorized by the number of frequencies processed in the pipeline:

```plaintext
├── experimental_result_data/ # Spreadsheet diagrams of experimental results
├── polynomial_timing/ 	      #CPU and GPU tool for timing calcultaion
├── simsdp-g2g-imaging-pipeline/
    ├── Algo/        # Dataflow Algorithms for pipeline steps 
    ├── Archi/       # HPC S-LAM Architecture models (e.g., HPC topology, node configurations)
    ├── Code/        # Source and generated code
    ├── Scenarios/   # Simulation scenarios 
    ├── Workflows/   # Workflow descriptions of the HPC resource allocation and simulation
├── simsdp-fft-imaging-pipeline/
├── simsdp-dft-imaging-pipeline/
├── simsdp-g2g-imaging-pipeline_nfreq/
├── simsdp-fft-imaging-pipeline_nfreq/
├── simsdp-dft-imaging-pipeline_nfreq/
```

> [!NOTE]
>
> The original Generic Imaging Pipeline has been formatted with the automatic cycle-unrolling integrated into PREESM to respect the dataflow paradigm in order to fit input requirement of the dataflow-based SimSDP framework.



---

## Key Features

- **HPC Resource Allocation**:  
  Implements a model-based approach to optimize workload distribution between nodes using **PREESM**.
  
- **Simulation of Large Datasets**:  
  Handles simulations for datasets as large as those required by the SKA (xTB).
  
- **Modular Pipeline Design**:  
  Each pipeline step (G2G, FFT, DFT) is modular, supporting easy adaptation to different architectures and use cases.
  
- **Single and Multi-Frequency Support**:  
  Simulates pipelines for both single and multi-frequency imaging scenarios.

---

## Requirements

- **PREESM** framework ([installation instructions](https://preesm.github.io/get/)).
- SimSDP may not be full integrated in the last PREESM release, to access it clone the [Preesm - github repository](https://github.com/preesm/preesm/) and branch `clustering2`.
- Access to an HPC environment.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone git@gitlab-research.centralesupelec.fr:dark-era/simsdp-generic-imaging-pipeline.git
   cd simsdp-generic-imaging-pipeline

2. Launch **PREESM** and open folders from **PREESM**: File>open project from file system> browse `simsdp_g2g_imaging_pipeline` folder.

### Parameterized timing estimation
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>
Timings definition consist in polynomials calculation, the procedure is the following. For each dataflow pipeline configuration do:
This section consist in setting up a method to define actor timings with a fitting function to facilitate algorithm comparison varying parameters. The method consist in building sampling (stored in /averages folder) and compute fitting function for each actor. Here are the instruction for running the method build by Sunrise Wang which a manual method evaluating few samples of data. The automated method extending Sunrise\'s work can be found in [polynomials_timing](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/tree/main/polynomial_timing?ref_type=heads) folder.


1. Build the the benchmark that will be stored in [averages](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/tree/main/polynomial_timing/averages?ref_type=heads) folder:
   1. In **PREESM**, the timings algorithm are provided in the algo/ folder select for example `grid_timing.diagram` and tune the parameter values: `NUM_VIS` [1000000; 2000000;3000000;4000000], `GRID_SIZE` [65536; 262144; 589824; 1048576; 1638400; 2359296; 3211264; 4194304]; , `NUM_MINOR_CYCLE`.
   2. Open `timing.scenario` and check that the algorithm is `grid_timing.pi`
   3. Generated the code: click on `codegen2.worklow` > `run workflow` > select `timing.scenario. This has to be done for each configuration.
   
2. Run the code: `cd Code` > `cmake.` > `make` > `./project_name`, a file with the computed timing is generated in the folder.

3. For each configuration concatenate the file such as : `time acquisition1; GRID_SIZE1; NUM_VISIBILITY1;time acquisition2; GRID_SIZE2; NUM_VISIBILITY2;...`

4. Executing the `plot_and_fit_averages.py` script to obtain the fit function. Install scipy `pip install scipy`. Execute : `python plot_and_fit_averages.py <input_file> <num_axis[1-2]> <dof[1-2]> <num_x_datapoints[1]> <num_y_datapoints [file_len/3]>` where the two last parameter are used for deconv,degridding etc. ex `python plot_and_fit_averages.py degrid.csv 2 1 8 4`(on the benchmark there is 8 GRID_SIZE and 4 NUM_VIS).

   <div style="text-align: center;">
       <img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/Figure_1.png" alt="Description alternative" style="max-width: 80%;">
       <p><b>Figure 1:</b> 
       <pre><code>python plot_and_fit_averages.py degrid.csv 2 1 8 4</code></pre>  
   	RMSE: 2929.136239024687; <br>
      Polynomials: [-7.57870456e+02  1.11207625e-03  4.67772660e-04]
      </p>
   </div>

   <div style="text-align: center;">
       <img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/Figure_2.png" alt="Description alternative" style="max-width: 80%;">
       <p><b>Figure 2:</b> 
   	In the original GIP paper the retain value are:
       <pre><code>python plot_and_fit_averages.py degrid.csv 2 2 8 4</code></pre>   
   	RMSE: 1310.353129574978; <br>
    Polynomials:  [-5.38868774e+02  8.93032173e-04 -1.99593895e-11  9.06752993e-04
     1.90789332e-10 -2.23468265e-10]
      </p>
   </div>

   <div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/polynomial_timing/Figure_3.png" alt="Description alternative" style="max-width: 80%;">
    <p><b>Figure 3:</b> 
   The dof value that offer the best RMSE for 8x4 input parameter is 6:
    <pre><code>python plot_and_fit_averages.py degrid.csv 2 6 8 4</code></pre>  
   RMSE: 254.3481181405604; <br>
   Polynomials: [ 5.31713245e+04 -8.30510174e-02  2.42638662e-08  1.12866458e-14
    -5.01452759e-21  3.07430467e-31  1.10957450e-34 -2.16795144e-02
     3.83814482e-08 -2.25243447e-14  4.15817783e-21  2.80271118e-28
    -1.08010603e-34  8.34323283e-09 -5.89982326e-15  4.00900882e-21
    -1.06617401e-27  1.09531005e-34 -5.42048113e-15 -2.23817508e-22
    -1.06094371e-28 -3.91299875e-36  2.55420177e-21  1.59977576e-28
     1.48860113e-35 -5.80263610e-28 -2.24210276e-35  5.00611356e-35]
   </p>
   </div>


   > The axis represent: 
   >
   > - z:  the execution time for each configuration.
   >
   > - x: the fisrt parameter, here GRID_SIZE.
   >
   > - y: the second parameter, here NUM_VIS
   >
   > For each Figure:
   >
   > - The upper plot represent the measured data for each x,y value.
   >
   > - The bottom plot represent the polynomial model of the fitting function.

We target a Root Mean Square Error (RMSE) as short as possible while minimizing the number of coefficient , the value are the coefficient of the polynomials. 
The number of coefficients has an impact on the RMSE and must be less than the number of points acquired, otherwise it crashes, you should respect the following:

	num_coeffs = (dof + 1)(dof + 2) / 2 ≤ num_points
hence:

	dof ≤ (-3 + sqrt(9 + 8 × num_points)) / 2

</details>

### Run SimSDP on a defined HPC architecture

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. Create a CSV file that will be exploit by the method to generate  multinode multicore HPC S-LAM files. Name this file “SimSDP_archi.csv”  and save it in the **Archi** folder. Your file should look like this:    

   ```bash
   Node name;Core ID;Core frequency;Intranode rate;Internode rate
   Node0;0;2000.0;472.0;9.42
   Node0;1;2000.0;472.0;9.42
   Node0;2;2000.0;472.0;9.42
   Node0;3;2000.0;472.0;9.42
   Node1;0;2000.0;472.0;9.42
   Node1;1;2000.0;472.0;9.42
   Node1;2;2000.0;472.0;9.42
   Node1;3;2000.0;472.0;9.42
   Node2;0;2000.0;472.0;9.42
   Node2;1;2000.0;472.0;9.42
   Node2;2;2000.0;472.0;9.42
   Node2;3;2000.0;472.0;9.42
   ```

2. Configure your network architecture: right click on your project “Preesm > generate custom architecture network”. Choose the network you  want. You can let it as default. It generates a XML file stored in the **Archi** folder (you can update it as you want). Your file should look like this:  
    ```html
   <!-- Cluster with shared backbone:3:4:1 -->
   <?xml version='1.0'?>
   
   <!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
   <platform version="4.1">
   <zone id="my zone" routing="Floyd">
   <cluster id="Dragonfly cluster" prefix="Node" radical="0-2" suffix="" speed="1f" bw="125MBps" lat="50us" topology="DRAGONFLY" topo_parameters="1,3;1,2;1,1;3">
         <prop id="wattage_per_state" value="90.0:90.0:150.0" />
         <prop id="wattage_range" value="100.0:200.0" />
     </cluster>
   </zone>
   </platform>
   ```
3. Run SimSDP: Right-click on the workflow “/Workflows/**hypervisor**.workflow” and select “Preesm > Run Workflow”, In the scenario selection wizard, select “/Scenarios/init_**.scenario

During its execution, the workflow will log information into the  Console of Preesm. When running a workflow, you should always check this console for warnings and errors (or any other useful information).

Additionnaly, the workflow execution generates intermediary dataflow graphs that can be found in the **/Algo/generated/** directory. The C code generated by the workflow is contained in the **/Code/generated/** directory. The simulated data are stored in the **/Simulation** directory.

4. A python notebook is provided in the SimSDP project to analyse the simulator generated files: Launch `jupyter notebook` and open “SimSDPproject/SimulationAnalysis.ipynb”. Make sure that the  CSVs are in the reading path. Load each code to display the trends with  your simulated data.
   </details>
   
### Run SimSDP on a defined range of architectures
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. Add a SimSDP_moldable.csv in the **Archi** folder  which should look something like this:

 | Parameters       | min        | max        | step       |
   | ---------------- | ---------- | ---------- | ---------- |
   | number of nodes  | 1          | 6          | 1          |
   | number of cores  | 1          | 6          | 1          |
   | core frequency   | 1          | 1          | 1          |
   | network topology | 1          | 4          | 1          |
   | node memory      | 1000000000 | 1000000000 | 1000000000 |

2. Run SimSDP: Right-click on the workflow “/Workflows/**hypervisor_dse**.workflow” and select “Preesm > Run Workflow”, In the scenario selection wizard, select “/Scenarios/init_**.scenario
3. Analyse the simulation with the`jupyter notebook` .
</details>

### Run the generated code
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. install the requirements:
```
sudo apt-get install libfftw3-dev

#BLAS
sudo apt-get install libblas-dev

#LAPACK
sudo apt-get install liblapack-dev
sudo apt-get install liblapacke-dev

#notebook to visualize
sudo apt-get install python3-pip
sudo apt install jupyter-notebook

#ASTROPY
sudo apt install python3-astropy
 
check: python3 -c "import astropy; print(astropy.__version__)"
```
2. Download [GLEAM](https://nasext-vaader.insa-rennes.fr/ietr-vaader/preesm/assets/sep_data.zip) (or whatever dataset).

3. copy past the data in Code/data/ folder. If it doesn't exist create a folder output/small/ inside.

4. Run the code and wait till your prompt display: `Process finished with exit code 0`.
    (It could be long depending on the **NUM_MAJOR_CYCLE** and the **NUM_MINOR_CYCLE**).

5. On CLion, for the CPU version, run the CMakeList.txt, build :hammer: and Run  the code :arrow_forward:.
  - Still on CLion, for GPU version, configure CMake:

    - install nvcc `sudo apt install nvidia-cuda-toolkit`, check the install `nvcc --version`.

    - Settings :gear:>Build, Execution, Deployment > CMake, add profile :heavy_plus_sign:, name `GIP_GPU`, CMake option `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` (if you use the emulator: option `-DUSE_CUDA_EMULATOR=ON`, the emulator only allows you to check that the code is functional, execution will be slower than on a GPU).


#### Understanding the dataset
Open a terminal and launch `jupyter notebook`
and retrieve the file : /ska_sep_preesm/notebooks/dataset_prep.ipynb

Reload each code to update with your setup (you have to wait some time on several display, don't panic).

#### Visualizing the outut
At this stage verify that your output folder contains files such as :"cycle_0_clean_psf.csv"

```
#Convert CSV files into fits files
python3 csvtoimage_all.py output/small/ fits/ ,

#install ds9
sudo apt install saods9

#Run
ds9 *.fits -lock frame wcs -zoom to fit
```
Filter: `/path to sep/ska_sep_preesm/Code/data/fits/*fits`

To reveal the contrasts:

- Color > Matplotlib > turbo (recommended by Sunrise)
- Color > Matplotlib > viridis / inferno (most popular in astro-papers)

![](https://github.com/Ophelie-Renaud/Imaging/blob/main/DS9_g2g_example1.png?raw=true)
    
</details>

## Experimental results

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

- Simulating generic imaging pipelines - 1 freq - CPU - balanced-workload based node partitioning - unoptimized code - [nVis = 3924480 , GRID_SIZE = 2048, nMinorCycle = 200] :

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/1freq.png)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [**nVis** = 10xNUM_BASELINE:5xNUM_BASELINE:30xNUM_BASELINE , GRID_SIZE = 2048, nMinorCycle = 200]:

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/simu_nvis.png)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [nVis = 3924480 , **GRID_SIZE** = 512:512:2560, nMinorCycle = 200]:

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/simu_grid.png)

- Simulating generic imaging pipelines - 21 freq - CPU - frequency-based node partitioning - [nVis = 3924480 , GRID_SIZE = 2048, **nMinorCycle** = 50:50:250]:

![](https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/simu_minor.png)



- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [**nVis** = 1:784896:3924480 , nKernel = 108800, nMinorCycle = 200]:

[ToDo]

- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [nVis = 3924480 , **nKernel** = 1:21760:108800, nMinorCycle = 200]:

[ToDo]

- Simulating generic imaging pipelines - 21 freq - **GPU** - frequency-based node partitioning - [nVis = 3924480 , nKernel = 108800, **nMinorCycle** = 1:40:200]:

[ToDo]
</details>

## Citation

If you use this work, please cite the following publication:

```plaintext
# This publication for the Generic Imaging pipeline and the manual version of fitting function calculation
@InProceedings{10.1007/978-3-031-62874-0_5,
author="Wang, Sunrise
and Gac, Nicolas
and Miomandre, Hugo
and Nezan, Jean-Francois
and Desnos, Karol
and Orieux, Francois",
editor="Dias, Tiago
and Busia, Paola",
title="An Initial Framework for Prototyping Radio-Interferometric Imaging Pipelines",
booktitle="Design and Architectures for Signal and Image Processing",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="56--67"
}

# This publication for the design space exploration method
@inproceedings{renaud:hal-04608249,
  TITLE = {{Multicore and Network Topology Codesign for Pareto-Optimal Multinode Architecture}},
  AUTHOR = {Renaud, Oph{\'e}lie and Desnos, Karol and Raffin, Erwan and Nezan, Jean-Fran{\c c}ois},
  URL = {https://hal.science/hal-04608249},
  BOOKTITLE = {{EUSIPCO}},
  ADDRESS = {Lyon, France},
  ORGANIZATION = {{EURASIP}},
  PUBLISHER = {{IEEE}},
  PAGES = {701-705},
  YEAR = {2024},
  MONTH = Aug,
  DOI = {10.23919/EUSIPCO63174.2024.10715023},
  KEYWORDS = {HPC Simulation ; Dataflow model of computation ; Design space exploration DSE ; Network topology},
  PDF = {https://hal.science/hal-04608249v1/file/EUSIPCO_Multicore_and_Network_Topology_Codesign_for_Pareto_Optimal_Multinode_Architecture-2.pdf},
  HAL_ID = {hal-04608249},
  HAL_VERSION = {v1},
}

# Soon the SimSDP resource allocation method
# Soon the SimSDP proof of concept on HPC system with automated fitting function
```

## Contact  

For questions or feedback, please contact:  
- [Ophélie Renaud](mailto:ophelie.renaud@ens-paris-saclay.fr)

## Acknowledgement

*This work was supported by DARK-ERA (ANR-20-CE46-0001-01).*
