# SimSDP - Generic Imaging Pipeline

SimSDP is an HPC resource allocation framework and simulation tool designed to optimize intra-node and inter-node resource allocation for scientific data processing. It combines **PREESM** for rapid prototyping and **SimGrid** for simulating inter-node communication and computation. 

The primary goal of SimSDP is to simulate the **Science Data Processor (SDP)** pipeline from the Square Kilometre Array (SKA), converting visibilities into output images on massive HPC architectures, handling datasets as large as the SKA requirements (xTB).


The aim of this project is to facilitate the comparison of algorithm describing the SDP using SimSDP.  As a dataflow-based tool, SimSDP takes into account the dataflow application graph as well as a csv file containing the execution times of the actors on each target to be simulated. Dataflow representation of  the **Generic Imaging Pipeline** with a single and multi-frequency scenarii are provided. Our goal is to simulate these pipelines on Multinode - Multicore architecture, on Multinode - MonoGPU architecture and (might be) on Multinode - MultiGPU architecture.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal.png" style="zoom:100%;" />

Algorithm performance is influenced by parameters such as the number of visibility points, grid size and number of minor cycles. We aim to identify and evaluate the influence of these parameters on scaling and assess whether a solution is viable in the context of SKA.

To achieve this, we provide tools that automatically estimate the execution time of dataflow actors for each targeted architecture and simulate algorithm performance by varying a set of parameter values automatically.

<img src="https://raw.githubusercontent.com/Ophelie-Renaud/simsdp-generic-imaging-pipeline/refs/heads/main/experimental_result_data/project_goal2.png" style="zoom:30%;" />

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

Clone the repository:
```bash
git clone git@gitlab-research.centralesupelec.fr:dark-era/simsdp-generic-imaging-pipeline.git
cd simsdp-generic-imaging-pipeline
```
---

### Polynomial regression for static timing estimation
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

This section consist in setting up a method to define actor timings with a fitting function to facilitate algorithm comparison varying parameters. The method consist in building sampling (stored in **/averages** :file_folder:) and compute fitting function for each actor. The original method was setting up by Sunrise Wang and consist in a manual method evaluating few samples of data (details of the method are available in the [wiki](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/wikis/pages)), however once benchmark is set up additional instruction can be found in [polynomials_timing](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/tree/main/polynomial_timing?ref_type=heads) :file_folder: section **SOTA**. The proposed automated method extending Sunrise\'s work can be found in [polynomials_timing](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/tree/main/polynomial_timing?ref_type=heads) :file_folder: section **Proposed method**.  
</details>

---

### Optimized G2G integration
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. generate the *.so librarie: `cd g2g_lib` > `cmake .` > `make`, the lib.so will be built in **build** :file_folder:.

2. create `libcpu_skytosky_single.h`:
```c
#ifndef LIBCPU_SKYTOSKY_SINGLE_H
#define LIBCPU_SKYTOSKY_SINGLE_H

#ifdef __cplusplus
extern "C" {
#endif

// Déclarations des fonctions exportées
void degridding_quad_pola(void);
void dgg_init_s2s(void);
void free_params(void);
void gridding_psf(void);
void gridding_quad_pola(void);
void init(void);
void s2s_quad_pola(void);
void s2s_single_pola(void);
void get_sky2sky_matrix_v0(struct interpolation_parameters* params);
void get_sky2sky_matrix_v1(struct interpolation_parameters* params);
void get_sky2sky_matrix_v3(struct interpolation_parameters* params);

#ifdef __cplusplus
}
#endif

#endif // LIBCPU_SKYTOSKY_SINGLE_H
```
3. include header: `#include "libcpu_skytosky_single.h"`.

4. compile with the lib: `gcc -o exe main.c -libcpu_skytosky_single.so -lcpu_skytosky_single -libcpu_skytosky_single.h`.

   > This is how optimized G2G has been include in our computation set.
</details>

---

### Run SimSDP on a simulated HPC architecture

<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

The SimSDP consist in 3 main steps:

- **Node-level partitioning**: Divide the dataflow graph into subgraph, each associated to an architecture node.
- **Thread-level partitioning**: For each subgraph allocate resources on an architecture node.
- **Simulation**: Simulate the intra- and inter- architecture node behavior.


SimSDP has been [is going to be] updated in order to manage several partitioning mode.


- <u>**Manual mode** (this project): One subgraph on the topgraph is associated to a node architecture</u>.
- **Random mode**: The whole graph is partitioned among the available node and distributed in random-workload.
- **Balanced workload mode** (the original method, for more detail see [wiki](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/wikis/pages)): The whole graph is partitioned among the available node and distributed in balanced-workload.

#### Simulating on <u>multicore</u> & multinode architecture

Setting up the manual mode: open preesm projects > `workflows/NodePartitioning.worflow` > select the `NodePartitioner` task > `properties` > `Partitioning mode` :arrow_right: `manual`.

##### Single node simulation

1. Launch **PREESM** and open defined preesm projects: `File` > `open project from file system` > browse: `simsdp_g2g_imaging_pipeline` folder.

2. Run simulation: `Workflows/codegen.worflow` > right click > `Preesm` > `run workflow` > browse: `1` or `4core.scenario`.

   During its compilation :hourglass:, the workflow will log information into the Console of Preesm. When running a workflow, you should always check this console for warnings and errors (or any other useful information).

   The C code generated by the workflow is contained in the **/Code/generated/** directory. 

3. Result :bar_chart: : A Gantt chart is generated.

##### multi node simulation

1. Launch **PREESM** and open defined preesm projects: `File `> `open project from file system` > browse: `simsdp_g2g_imaging_pipeline_nfreq` folder.

2. Run simulation: `Workflows/codegen.worflow` > right click > `Preesm` > `run workflow` > browse: `hypervisor.workflow` .

   During its compilation :hourglass:, the workflow will log information into the   Console of Preesm. When running a workflow, you should always check this console for warnings and errors (or any other useful information).

   Additionnaly, the workflow execution generates intermediary dataflow graphs that can be found in the **/Algo/generated/** directory. The C code generated by the workflow is contained in the **/Code/generated/** directory. The simulated data are stored in the **/Simulation** directory.

3. Result :bar_chart: : A python notebook is provided in the SimSDP project to analyse the simulator generated files: Launch `jupyter notebook` and open *SimSDPproject/SimulationAnalysis.ipynb*. Make sure that the  CSVs are in the reading path. Load each code to display the trends with  your simulated data.

</details>

---

### Run the generated code
<details>
    <summary style="cursor: pointer; color: #007bff;"> Click here to reveal the section </summary>

1. install the requirements:

```bash
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

---

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
```

```plaintext
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
```

```plaintext
# Soon the SimSDP resource allocation method
```

```plaintext
# Soon the SimSDP proof of concept on HPC system with automated fitting function
```

## Wiki

For references and so on take a look on the [wiki](https://gitlab-research.centralesupelec.fr/dark-era/simsdp-generic-imaging-pipeline/-/wikis/pages).

## Contact  

For questions or feedback, please contact:  
- [Ophélie Renaud](mailto:ophelie.renaud@ens-paris-saclay.fr)

## Acknowledgement

*This work was supported by DARK-ERA (ANR-20-CE46-0001-01).*
